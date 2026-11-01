import copy
import os
import torch
import pandas as pd
import collections
import pickle
import json
import math
import re
from tqdm import tqdm

from argparser import parse_args
from utils import set_seed, prepare_model
from config import load_config
from datasets import load_data
from ver_utils import get_ovalbab_network, run_oval_bab, create_1_vs_all_verification_problem
from attack import pgd_attack
from certified import get_C
from utils import compute_perturbation
from auto_LiRPA.utils import logger
from auto_LiRPA import BoundedModule


logging_interval = 100


def report(args, pbar, tot_verified_corr, tot_crown_verified, tot_ibp_verified, tot_nat_ok, tot_pgd_ok,
           test_idx, tot_tests, current_logs):
    """ Logs evaluation statistics to standard output. """
    pbar.set_description(
        'tot_tests: %d, oval_ver_ok: %.5lf [%d/%d], oval_nat_ok: %.5lf [%d/%d], oval_pgd_ok: %.5lf [%d/%d]' % (
            tot_tests,
            tot_verified_corr/tot_tests, tot_verified_corr, tot_tests,
            tot_nat_ok/tot_tests, tot_nat_ok, tot_tests,
            tot_pgd_ok/tot_tests, tot_pgd_ok, tot_tests,
        )
    )

    if (test_idx + 1) % logging_interval == 0:
        prefix = 'oval' if not args.crown_ver else 'crown'
        verification_dict = {
            f'{prefix}_ver_ok': tot_verified_corr/tot_tests,
            'ibp_ver_ok': tot_ibp_verified/tot_tests,
            f'{prefix}_nat_ok': tot_nat_ok/tot_tests,
            f'{prefix}_pgd_ok': tot_pgd_ok/tot_tests,
        }
        if not args.crown_ver:
            verification_dict['crown_ver_ok'] = tot_crown_verified/tot_tests
        for ckey in verification_dict:
            current_logs[test_idx][ckey] = verification_dict[ckey]
        current_logs[test_idx]['idx'] = test_idx


def main(args):

    num_eps = 1e-3
    if not os.path.exists("verification_logs"):
        os.makedirs("verification_logs")
    current_logs = collections.defaultdict(dict)  # per-epoch logs + summary

    if torch.cuda.is_available():
        # Disable the 19-bit TF32 type, which is not precise enough for verification purposes
        # see https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        if float(torch.__version__.split(".")[0]) >= 2:
            # if torch >= 2.0
            # https://docs.pytorch.org/docs/stable/notes/cuda.html
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    config = load_config(args.config)
    bound_config = config['bound_params']
    logger.info('config: {}'.format(json.dumps(config)))

    # Set random seed. If there was a seed in the model name, override anything from args or config.
    regexp = re.search(r'seed_\d', args.load)
    if regexp:
        args.seed = int(regexp.group(0).split('_')[1])
    set_seed(args.seed or config['seed'])

    # Load dataset and network.
    model_ori, checkpoint, epoch = prepare_model(args, logger, config)
    model_ori.eval()
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = 1
    dummy_input, _, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)

    name = f"V_{os.path.basename(args.load)}"
    if args.start_idx != 0:
        name += f"_s:{args.start_idx}"
    if args.end_idx != -1:
        name += f"_e:{args.end_idx}"

    if args.pickle_results:
        # Prepare pickle for results.
        if not os.path.exists(args.pickle_results):
            os.makedirs(args.pickle_results)
        json_name = os.path.basename(args.oval_bab_config.replace(".json", ""))
        record_name = args.pickle_results + os.path.basename(
            args.load.replace("/", "_").replace(".", "")) + "_" + json_name + ".pkl"
        if os.path.isfile(record_name):
            results_table = pd.read_pickle(record_name)
        else:
            dataset_len = args.end_idx if args.end_idx != -1 else len(test_data)
            indices = list(range(dataset_len))
            results_table = pd.DataFrame(
                index=indices, columns=["prop"] + [f'BSAT_{json_name}', f'BBran_{json_name}', f'BTime_{json_name}'])
            results_table.to_pickle(record_name)
    else:
        results_table, json_name, record_name = None, None, None

    # Convert net for OVAL BaB use.
    if not args.crown_ver:
        with torch.no_grad():
            torch_verif_layers = get_ovalbab_network(dummy_input, model_ori)

    # get autolirpa model to attempt quick certification before BaB
    model_lirpa = BoundedModule(
        copy.deepcopy(model_ori), dummy_input, bound_opts=config['bound_params']['bound_opts'], custom_ops={},
        device=args.device)
    model_lirpa.eval()

    eps = args.eps or bound_config['eps']
    data_max, data_min, std = test_data.data_max, test_data.data_min, test_data.std

    pbar = tqdm(test_data, dynamic_ncols=True)
    tot_verified, tot_crown_verified, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests = 0, 0, 0, 0, 0, 0
    test_idx = 0
    for test_idx, (inputs, targets) in enumerate(pbar):
        if test_idx < args.start_idx or (args.end_idx != -1 and test_idx >= args.end_idx):
            continue

        if results_table is not None:
            if pd.isna(results_table.loc[test_idx]["prop"]) == False:
                print(f'the {test_idx}th element is done')
                continue

        tot_tests += 1

        # Standard accuracy.
        nat_outs = model_lirpa(inputs.cuda()).cpu()
        nat_ok = targets.eq(nat_outs.max(dim=1)[1]).item()

        # Logging.
        tot_nat_ok += nat_ok
        if not nat_ok:
            report(args, pbar, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, current_logs)
            continue

        # Prepare to use OVAL BaB for verification
        assert inputs.shape[0] == 1, "only test_batch=1 is supported for OVAL BaB"
        # Compute the perturbation
        norm_eps = eps
        if type(norm_eps) == float:
            norm_eps = (norm_eps / std).view(1, -1, 1, 1)
        else:  # [batch_size, channels]
            norm_eps = (norm_eps.view(*norm_eps.shape, 1, 1) / std.view(1, -1, 1, 1))
        data_ub = torch.min(inputs + norm_eps, data_max)
        data_lb = torch.max(inputs - norm_eps, data_min)
        torch_input_bounds = torch.stack([data_lb, data_ub], dim=-1)
        torch_input_bounds = torch_input_bounds.squeeze(0)
        torch_targets = targets.squeeze(0)

        # Run a quick attack before BaB
        with torch.no_grad():
            adv_data = pgd_attack(
                model_lirpa, data_lb.cuda(), data_ub.cuda(),
                lambda x: torch.nn.CrossEntropyLoss(reduction='none')(x, targets.cuda()),
                args.test_att_n_steps, args.test_att_step_size)
            adv_outs = model_lirpa(adv_data.cuda()).cpu()
            adv_ok = targets.eq(adv_outs.max(dim=1)[1]).item()
        if not adv_ok:
            print("==========> A quick PGD attack found a vulnerability (no BaB needed)")
            report(args, pbar, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, current_logs)
            continue

        # Check whether the best of IBP/CROWN bounds suffice to prove the property (computes a bound per logit
        # difference, rather than one-vs-all as BaB)
        c = get_C(args, inputs.to(args.device), targets.to(args.device))
        x, data_lb, data_ub = compute_perturbation(
            args, eps, inputs.to(args.device), data_min.to(args.device), data_max.to(args.device), std.to(args.device),
            True, False)
        ibplb, _ = model_lirpa(x=(x,), method_opt="compute_bounds", IBP=True, C=c, method=None, no_replicas=True)
        if args.lirpa_crown_batch is not None:
            # compute the bounds in batches to save memory
            lb = -torch.ones((1, args.num_class - 1), device=c.device)
            n_batches = int(math.ceil(args.num_class / float(args.lirpa_crown_batch)))
            for sub_batch_idx in range(n_batches):
                # compute intermediate bounds on sub-batch
                start_batch_index = sub_batch_idx * args.lirpa_crown_batch
                end_batch_index = min((sub_batch_idx + 1) * args.lirpa_crown_batch, args.num_class)
                lb[:, start_batch_index:end_batch_index], _ = model_lirpa.compute_bounds(
                    x=(x,), IBP=False, C=c[:, start_batch_index:end_batch_index, :], method='CROWN',
                    bound_lower=True, bound_upper=False)
        else:
            lb, _ = model_lirpa.compute_bounds(
                x=(x,), IBP=False, C=c, method='CROWN', bound_lower=True, bound_upper=False)
        lb = torch.max(lb, ibplb)

        if args.debug and not args.crown_ver:
            # DEBUG: check if OVAL-based and auto_LiRPA based bounds match
            from plnn.naive_approximation import NaiveNetwork
            from plnn.proxlp_solver.propagation import Propagation
            with torch.no_grad():
                verif_domain = torch_input_bounds.cuda().unsqueeze(0)
                intermediate_net = NaiveNetwork([copy.deepcopy(lay).cuda() for lay in torch_verif_layers])
                intermediate_net.define_linear_approximation(verif_domain, override_numerical_errors=False, cdebug=True)
                o_lbs = intermediate_net.lower_bounds[-1].squeeze(0).cpu()
                d_lbs, _ = model_lirpa(x=(x,), method_opt="compute_bounds", IBP=True, method=None, no_replicas=True)
                # bounds should coincide despite of different auto_lirpa handling of BN because IBP bounds of a series
                # of linear layers don't get looser when adding an elementwise operation
                assert (o_lbs - d_lbs.cpu()).abs().max() < 1e-2
                prop_net = Propagation([copy.deepcopy(lay).cuda() for lay in torch_verif_layers], type='alpha-crown')
                prop_net.build_model_using_bounds(verif_domain, (intermediate_net.lower_bounds,
                                                                 intermediate_net.upper_bounds), cdebug=True)

        ibp_verified = ibplb.min() > 0
        tot_ibp_verified += int(ibp_verified)
        verified = lb.min() > 0
        tot_crown_verified += int(verified)
        if verified:
            tot_pgd_ok += 1.
            tot_verified += 1.
            print("==========> CROWN verified the property (no BaB needed)")
            report(args, pbar, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, current_logs)
            continue

        if args.crown_ver:
            tot_pgd_ok += 1.
            print("==========> CROWN didn't verify the property")
            report(args, pbar, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, current_logs)
            continue

        # Make problem a 1vsall robustness verification problem
        with torch.no_grad():
            torch_verif_problem = create_1_vs_all_verification_problem(
                torch_verif_layers, torch_targets, torch_input_bounds, args.ib_batch_size, inputs, c,
                use_lbs=args.oval_trim_logitdiff, lbs=lb, crown_1vsall=args.oval_crown_1vsall)
        # release some memory
        torch.cuda.empty_cache()

        with torch.no_grad():
            verified, pgd_ok = run_oval_bab(
                torch_verif_problem, torch_input_bounds, args.oval_bab_config, timeout=args.oval_bab_timeout,
                results_table=results_table, test_idx=test_idx, json_name=json_name, record_name=record_name)

        tot_pgd_ok += int(pgd_ok)
        tot_verified += int(verified)

        report(args, pbar, tot_verified, tot_crown_verified, tot_ibp_verified,
               tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, current_logs)

    current_logs["summary"]["model_dir"] = args.load
    current_logs["summary"]["host_name"] = os.uname().nodename
    last_idx = (args.end_idx - 1) if args.end_idx != -1 else test_idx
    if last_idx in current_logs:
        # add last logged idx to summary
        for ckey in current_logs[last_idx]:
            current_logs["summary"][ckey] = current_logs[last_idx][ckey]
    with open(f'verification_logs/{name}_logs.pickle', 'wb') as filehandle:
        pickle.dump(current_logs, filehandle)


if __name__ == '__main__':

    args = parse_args()
    main(args)
