import random

import torch
import re
import collections
import pickle
from auto_LiRPA import BoundedModule, CrossEntropyWrapper, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.bound_ops import *
from config import load_config
from datasets import load_data
from utils import *
from manual_init import manual_init, kaiming_init
from argparser import parse_args
from certified import ub_robust_loss, get_crown_loss, get_C, fetch_distillation_features_bounds, get_latent_from_lirpa
from attack import pgd_attack
from regularization import compute_reg, compute_L1_reg, compute_ibpr_reg
from tqdm import tqdm

args = parse_args()

if not args.verify:
    set_file_handler(logger, args.dir)
logger.info('Arguments: {}'.format(args))


def epsilon_clipping(eps, eps_scheduler, args, train):
    if eps < args.min_eps:
        eps = args.min_eps
    if args.fix_eps or (not train):
        eps = eps_scheduler.get_max_eps()
    if args.natural:
        eps = 0.
    return eps


def train_or_test(model, model_ori, t, loader, eps_scheduler, opt, current_logs, lr_scheduler=None, teacher=None):
    # Function used both for training and testing purposes

    pure_ibp = (not args.ccibp) and (not args.mtlibp) and (not args.sabr) and (not args.expibp) and (not args.pure_adv) and (not args.ccdist) and (not args.sabrdist) and (not args.mtldist) and (not args.ibprdist)

    train = opt is not None
    meter = MultiAverageMeter()

    data_max, data_min, std = loader.data_max, loader.data_min, loader.std
    if args.device == 'cuda':
        data_min, data_max, std = data_min.cuda(), data_max.cuda(), std.cuda()

    if train:
        model_ori.train(); model.train(); eps_scheduler.train()
        eps_scheduler.step_epoch()
    else:
        model_ori.eval(); model.eval(); eps_scheduler.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    for i, (data, labels) in enumerate(pbar):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        bounding_algorithm = args.bounding_algorithm

        if train:
            eps *= args.train_eps_mul
            att_n_steps = args.train_att_n_steps
            att_step_size = args.train_att_step_size
        else:
            att_n_steps = args.test_att_n_steps
            att_step_size = args.test_att_step_size
        attack_eps = eps * args.attack_eps_factor

        eps = epsilon_clipping(eps, eps_scheduler, args, train)
        attack_eps = epsilon_clipping(attack_eps, eps_scheduler, args, train)

        reg = t <= args.num_reg_epochs

        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = 'natural' if (eps < 1e-50) else 'robust'
        robust = batch_method == 'robust'

        # labels = labels.to(torch.long)
        if args.device == 'cuda':
            data, labels = data.cuda().detach().requires_grad_(), labels.cuda()

        # NOTE: all forward passes should be carried out on the LiRPA model to avoid batch_norm stats mismatches
        run_adv = (att_n_steps is not None and att_n_steps > 0)
        if pure_ibp or batch_method == 'natural' or (not run_adv and not pure_ibp) or (not train) or args.ccdistclean_loss:
            # Compute regular cross-entropy loss
            output = model(data)
            regular_ce = ce_loss(output, labels)  # regular CrossEntropyLoss used for warming up
            regular_err = torch.sum(torch.argmax(output, dim=1) != labels).item() / data.size(0)

            if (args.ccdist and args.ccdistclean_loss):
                student_cleanlatent = get_latent_from_lirpa(model, earlier_latent=args.distillation_earlier_latent)
        else:
            regular_ce = 0.
            regular_err = 0.

        # Compute the perturbation
        # NOTE: at validation (train=false) these losses and errors are computed on the target epsilon
        x, data_lb, data_ub = compute_perturbation(args, eps, data, data_min, data_max, std, robust, reg)
        # Run a PGD attack
        if run_adv:

            if train:
                # attack perturbation with a possibly different epsilon
                _, attack_lb, attack_ub = compute_perturbation(
                    args, attack_eps, data, data_min, data_max, std, robust, reg)

                # set the network in eval mode before the attack
                model_ori.eval()
                model.eval()
            else:
                attack_lb = data_lb
                attack_ub = data_ub

            with torch.no_grad():
                adv_data = pgd_attack(
                    model, attack_lb, attack_ub,
                    lambda x: nn.CrossEntropyLoss(reduction='none')(x, labels), att_n_steps, att_step_size)
                del attack_lb, attack_ub  # save a bit of memory

            if train:
                # reset the network in train mode post-attack (the adversarial point is evaluated in train mode)
                model_ori.train()
                model.train()

            # NOTE: differently from SABR, running stats are updated with the adversarial example too
            # (not much difference)
            adv_output = model(adv_data)
            adv_loss = ce_loss(adv_output, labels)
            adv_err = torch.sum(torch.argmax(adv_output, dim=1) != labels).item() / data.size(0)

        else:
            adv_loss = regular_ce
            adv_err = regular_err
            adv_output = output

        # Upper bound on the robust loss (via IBP)
        # NOTE: when training, the bounding computation will use the BN statistics from the last forward pass: in
        # this case, from the adversarial points
        if (robust or reg or args.xiao_reg) and not args.pure_adv:

            if (not args.sabr and not args.sabrdist) or (not train):

                if (args.ccdist or args.ibprdist or args.mtldist) and not args.kl_dist_loss \
                        and not args.ccdist1_dist_loss:
                    student_latent = get_latent_from_lirpa(model, earlier_latent=args.distillation_earlier_latent)

                robust_loss, robust_err, lb = ub_robust_loss(
                    args, model, x, data, labels, meter=meter, bounding_algorithm=bounding_algorithm)
            else:
                sabr_x, sabr_center = compute_sabr_perturbation(
                    args, attack_eps, data, adv_data, data_min, data_max, std, robust, reg)
                robust_loss, robust_err, lb = ub_robust_loss(
                    args, model, sabr_x, sabr_center, labels, meter=meter, bounding_algorithm=bounding_algorithm)

        else:
            lb = robust_loss = robust_err = None

        update_meter(meter, regular_ce, robust_loss, adv_loss, regular_err, robust_err, adv_err, data.size(0))

        if train:

            if reg and args.reg_lambda > 0:
                # the addition of (Shi et al. 2021)'s regularization appears to increase standard and
                # adversarial accuracy without significantly impacting training
                loss = compute_reg(args, model, meter, eps, eps_scheduler)
            else:
                loss = torch.tensor(0.).to(args.device)
            if args.l1_coeff > 0:
                loss += compute_L1_reg(args, model_ori, meter)

            if (not args.ccibp) and (not args.mtlibp) and (not args.sabr) and (not args.expibp) and (not args.ibpr)\
                    and (not args.ccdist) and (not args.sabrdist) and (not args.mtldist) and (not args.ibprdist):

                if robust and not args.pure_adv:
                    loss += robust_loss
                elif args.pure_adv:
                    loss += adv_loss
                else:
                    # warmup phase
                    loss += regular_ce

            else:

                if robust:
                    if args.ccibp:
                        # cross_entropy of convex combination of IBP with natural/adversarial logits
                        adv_diff = torch.bmm(
                            get_C(args, data, labels),
                            adv_output.unsqueeze(-1)).squeeze(-1)
                        ccibp_diff = args.ccibp_coeff * lb + (1 - args.ccibp_coeff) * adv_diff
                        loss += get_crown_loss(ccibp_diff)
                    elif args.mtlibp:
                        mtlibp_loss = args.mtlibp_coeff * robust_loss + (1 - args.mtlibp_coeff) * adv_loss
                        loss += mtlibp_loss
                    elif args.expibp:
                        expibp_loss = robust_loss ** args.expibp_coeff * adv_loss ** (1 - args.expibp_coeff)
                        loss += expibp_loss

                    elif args.ccdist:

                        # standard CC-IBP loss on the student model
                        adv_diff = torch.bmm(
                            get_C(args, data, labels),
                            adv_output.unsqueeze(-1)).squeeze(-1)
                        ccibp_logit_diff = args.ccibp_coeff * lb + (1 - args.ccibp_coeff) * adv_diff

                        distillation_data = adv_data if args.distillation_from_adv_t_repr else data

                        # distillation losses aimed at learning good CC-IBP features:
                        if args.kl_dist_loss:
                            # logit-based distillation
                            # KL loss between the CC-IBP logit differences and the nat/adv teacher logit differences
                            # NOTE: worth keeping as a baseline for a paper ablation
                            with torch.no_grad():
                                teacher_out = teacher(distillation_data)
                                teacher_logit_diff = torch.bmm(
                                    get_C(args, data, labels),
                                    teacher_out.unsqueeze(-1)).squeeze(-1)
                            distillation_loss = kl(
                                F.log_softmax(-ccibp_logit_diff / args.kl_temp, dim=1),
                                F.softmax(-teacher_logit_diff.detach() / args.kl_temp, dim=1)
                            ) / (data.size(0) * args.kl_temp ** 2)

                        else:
                            # feature-based distillation
                            # l2 between the worst-case convex combination of IBP bounds and adv logits, and the
                            # output of the teacher model on the nat/adversarial point
                            (latent_lb, latent_ub), teacher_latent = fetch_distillation_features_bounds(
                                teacher, model, distillation_data, earlier_latent=args.distillation_earlier_latent)

                            if args.ccdist1_dist_loss:
                                # NOTE: another paper ablation - remove the CC component from the
                                # feature-space distillation loss
                                distillation_loss = torch.maximum(
                                    torch.pow(latent_lb - teacher_latent, 2),
                                    torch.pow(latent_ub - teacher_latent, 2)
                                ).mean()
                            elif args.ccdist0_dist_loss:
                                distillation_loss = torch.pow(student_latent - teacher_latent, 2).mean()
                            elif args.ccdistclean_loss:
                                # do distillation between clean network passes
                                distillation_loss = torch.pow(student_cleanlatent - teacher_latent, 2).mean()
                            else:
                                distillation_loss = torch.maximum(
                                    torch.pow(args.ccibp_coeff * latent_lb + (1 - args.ccibp_coeff) * student_latent
                                              - teacher_latent, 2),
                                    torch.pow(args.ccibp_coeff * latent_ub + (1 - args.ccibp_coeff) * student_latent
                                              - teacher_latent, 2)
                                ).mean()

                        meter.update('distillation_loss', distillation_loss, 1)  # 1 as already using the mean

                        loss += get_crown_loss(ccibp_logit_diff) + args.distillation_coeff * distillation_loss

                    elif args.sabrdist:

                        distillation_data = adv_data if args.distillation_from_adv_t_repr else data

                        # feature-based distillation
                        # MSE between the worst-case convex combination of IBP bounds and adv logits, and the
                        # output of the teacher model on the nat/adversarial point
                        (latent_lb, latent_ub), teacher_latent = fetch_distillation_features_bounds(
                            teacher, model, distillation_data, earlier_latent=args.distillation_earlier_latent)

                        # NOTE: the distillation is wrt the SABR bounds
                        distillation_loss = torch.maximum(
                            torch.pow(latent_lb - teacher_latent, 2),
                            torch.pow(latent_ub - teacher_latent, 2)
                        ).mean()

                        meter.update('distillation_loss', distillation_loss, 1)  # 1 as already using the mean

                        loss += robust_loss + args.distillation_coeff * distillation_loss

                    elif args.mtldist:
                        # doing MTL-IBP distillation with an alpha term anyway (it's a bound owing to the
                        # CC-IBP relationship)

                        distillation_data = adv_data if args.distillation_from_adv_t_repr else data

                        # MSE between the worst-case convex combination of IBP bounds and adv logits, and the
                        # output of the teacher model on the nat/adversarial point
                        (latent_lb, latent_ub), teacher_latent = fetch_distillation_features_bounds(
                            teacher, model, distillation_data, earlier_latent=args.distillation_earlier_latent)

                        # same distillation term as when doing it for CC-IBP
                        distillation_loss = torch.maximum(
                            torch.pow(args.mtlibp_coeff * latent_lb + (1 - args.mtlibp_coeff) * student_latent
                                      - teacher_latent, 2),
                            torch.pow(args.mtlibp_coeff * latent_ub + (1 - args.mtlibp_coeff) * student_latent
                                      - teacher_latent, 2)
                        ).mean()

                        meter.update('distillation_loss', distillation_loss, 1)  # 1 as already using the mean

                        loss += args.mtlibp_coeff * robust_loss + (1 - args.mtlibp_coeff) * adv_loss
                        loss += args.distillation_coeff * distillation_loss

                    elif args.ibpr or args.ibprdist:
                        loss += adv_loss
                        # IBP-R over areas of intermediate bounds
                        loss += compute_ibpr_reg(args, model, meter) / data.shape[0]

                        if args.ibprdist:
                            loss += adv_loss

                            distillation_data = adv_data if args.distillation_from_adv_t_repr else data
                            # MSE between the worst-case convex combination of IBP bounds and adv logits, and the
                            # output of the teacher model on the nat/adversarial point
                            _, teacher_latent = fetch_distillation_features_bounds(
                                teacher, model, distillation_data, earlier_latent=args.distillation_earlier_latent)

                            # NOTE: the distillation is using the student latent only
                            distillation_loss = torch.pow(student_latent - teacher_latent, 2).mean()

                            loss += args.distillation_coeff * distillation_loss

                    else:
                        # sabr
                        sabr_loss = robust_loss
                        loss += sabr_loss
                else:
                    # warmup phase
                    loss += regular_ce

            meter.update('Loss', loss.item(), data.size(0))

            loss.backward()

        if train:
            grad_norm = torch.nn.utils.clip_grad_norm_(model_ori.parameters(), max_norm=args.grad_norm)
            meter.update('grad_norm', grad_norm)
            opt.step()
            opt.zero_grad()
            if args.lr_step == "batch" and lr_scheduler is not None:
                lr_scheduler.step()

        meter.update('wnorm', get_weight_norm(model_ori))
        meter.update('Time' , time.time() - start)

        pbar.set_description(
            ('[T]' if train else '[V]') +
            ' epoch=%d, nat_loss=%.4f, nat_ok=%.4f, adv_ok=%.4f, ver_ok=%.4f, ver_loss=%.3e, eps=%.4f' % (
                t,
                meter.avg('CE'),
                1. - meter.avg('Err'),
                1. - meter.avg('Adv_Err'),
                1. - meter.avg('Rob_Err'),
                meter.avg('Rob_Loss'),
                eps
            )
        )

    if batch_method != 'natural':
        meter.update('eps', eps)

    if train:
        epoch_logs = {
            'train_nat_loss': meter.avg('CE'),
            'train_nat_ok': 1. - meter.avg('Err'),
            'train_adv_ok': 1. - meter.avg('Adv_Err'),
            'train_adv_loss': meter.avg('Adv_Loss'),
            'train_ver_ok': 1. - meter.avg('Rob_Err'),
            'train_ver_loss': meter.avg('Rob_Loss'),
        }
        if args.ccdist:
            epoch_logs['distillation_loss'] = meter.avg('distillation_loss')
        for ckey in epoch_logs:
            current_logs[t][ckey] = epoch_logs[ckey]
    else:
        epoch_logs = {
            'val_nat_loss': meter.avg('CE'),
            'val_nat_ok': 1. - meter.avg('Err'),
            'val_adv_ok': 1. - meter.avg('Adv_Err'),
            'val_adv_loss':  meter.avg('Adv_Loss'),
            'val_ver_ok': 1. - meter.avg('Rob_Err'),
            'val_ver_loss': meter.avg('Rob_Loss'),
        }
        for ckey in epoch_logs:
            current_logs[t][ckey] = epoch_logs[ckey]
    current_logs[t]['epoch'] = t
       
    return meter


def main(args):
    if torch.cuda.is_available():
        if not args.enable_train_tf32:
            # Disable the 19-bit TF32 type, which is not precise enough for verification purposes, and seems to hurt
            # performance a bit for training
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
        else:
            # explicitly enable TF32
            if float(torch.__version__.split(".")[0]) >= 2:
                # if torch >= 2.0
                torch.backends.fp32_precision = "tf32"
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
                torch.backends.cudnn.rnn.fp32_precision = "tf32"
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    if not os.path.exists("logs"):
        os.makedirs("logs")
    current_logs = collections.defaultdict(dict)  # per-epoch logs + summary

    config = load_config(args.config)
    logger.info('config: {}'.format(json.dumps(config)))

    # Set random seed. If there was a seed in the model name, override anything from args or config.
    regexp = re.search(r'seed_\d', args.load)
    if regexp:
        args.seed = int(regexp.group(0).split('_')[1])
    seed = args.seed or config['seed']
    set_seed(seed)

    model_ori, checkpoint, epoch = prepare_model(args, logger, config)
    logger.info('Model structure: \n {}'.format(str(model_ori)))
    timestamp = int(time.perf_counter_ns())  # nanoseconds to avoid time collisions in checkpointing

    if args.ccdist or args.sabrdist or args.mtldist or args.ibprdist:
        teacher_model = get_teacher_model(args, config)
        teacher_model.to(args.device)
    else:
        teacher_model = None

    custom_ops = {}
    bound_config = config['bound_params']
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = args.test_batch_size or batch_size
    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)
    bound_opts = bound_config['bound_opts']

    model_ori.train()
    model = BoundedModule(model_ori, dummy_input, bound_opts=bound_opts, custom_ops=custom_ops, device=args.device)
    model_ori.to(args.device)

    if args.ccdist or args.sabrdist or args.mtldist or args.ibprdist:
        assert len(model.output_name) == 1, "Distillation assumes that the LiRPA network has a single output"

    if checkpoint is None:
        if args.manual_init:
            manual_init(args, model_ori, model, train_data)
        if args.kaiming_init:
            kaiming_init(model_ori)

    model_loss = model
    params = list(model_ori.parameters())
    logger.info('Parameter shapes: {}'.format([p.shape for p in params]))
    if args.multi_gpu:
        raise NotImplementedError('Multi-GPU is not supported yet')

    opt = get_optimizer(args, params, checkpoint)
    max_eps = args.eps or bound_config['eps']
    eps_scheduler = get_eps_scheduler(args, max_eps, train_data)
    lr_steps = args.num_epochs * len(train_data)
    lr_scheduler = get_lr_scheduler(args, opt, lr_steps=lr_steps)
    train_method, alpha = naming_util(args)

    if checkpoint is None:
        if args.ccdist:
            second_term = f"cc:{args.ccibp_coeff}"
        elif args.sabrdist:
            second_term = f"sabr:{args.sabr_coeff}"
        elif args.mtldist:
            second_term = f"mtl:{args.mtlibp_coeff}"
        elif args.ibprdist:
            second_term = f"ibpr:{args.ibpr_coeff}"
        else:
            second_term = f"l1:{args.l1_coeff}"
        name_prefix = f"{args.model}_{dict_to_string(parse_opts(args.model_params))}_{train_method}_" \
                      f"alpha:{alpha}_{second_term}_eps:{max_eps:.4f}_seed_{seed}_{timestamp}_"
    else:
        name_prefix = f"{os.path.basename(args.load)}"
        name_prefix = name_prefix.split('ckpt')[0]

    if epoch > 0 and not args.plot and not args.verify:
        # when loading from checkpoint, sync schedulers to the correct epoch
        # do similarly for the train dataloader, which is shuffled
        eps_scheduler.train()
        for i in range(1, epoch + 1):
            if args.lr_step == "epoch":
                lr_scheduler.step()
            else:
                for _ in train_data:
                    lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=False)
            pbar = tqdm(train_data, dynamic_ncols=True)
            for _, (data, labels) in enumerate(pbar):
                pbar.set_description('[T]' + ' epoch=%d' % i)

    if args.checkpointing and checkpoint is not None:
        # load PRNG state from checkpoint
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_prng'])
        torch.set_rng_state(checkpoint['torch_prng'])

    if args.verify:
        t = 0
        start_time = time.time()
        logger.info('Inference')
        meter = train_or_test(model, model_ori, 10000, test_data, eps_scheduler, None, current_logs, None)
        logger.info(meter)
        timer = time.time() - start_time
    else:
        timer = 0.0
        t = 0
        for t in range(epoch + 1, args.num_epochs + 1):
            start_time = time.time()
            train_or_test(model, model_ori, t, train_data, eps_scheduler, opt, current_logs, lr_scheduler=lr_scheduler,
                          teacher=teacher_model)
            update_state_dict(model_ori, model_loss)
            epoch_time = time.time() - start_time
            timer += epoch_time
            if args.lr_step == "epoch":
                lr_scheduler.step()
            if t % args.test_interval == 0:
                # Validation phase (performed on the target epsilon)
                with torch.no_grad():
                    meter = train_or_test(model, model_ori, t, test_data, eps_scheduler, None, current_logs)
                if args.checkpointing:
                    save(args, name_prefix, epoch=t, model=model_ori, opt=opt, intermediate_checkpoint=True)
                    current_logs["summary"]["model_dir"] = os.path.join(args.dir, name_prefix)

        if args.checkpointing:
            intermediate_path = os.path.join(args.dir, name_prefix + 'ckpt_intermediate')
            try:
                os.remove(intermediate_path)
            except OSError:
                pass
        save(args, name_prefix, epoch=t, model=model_ori, opt=opt)

    current_logs["summary"]["runtime"] = timer
    current_logs["summary"]["model_dir"] = os.path.join(args.dir, name_prefix + 'ckpt_last')
    current_logs["summary"]["host_name"] = os.uname().nodename
    if t in current_logs:
        # add last epoch to summary
        for ckey in current_logs[t]:
            current_logs["summary"][ckey] = current_logs[t][ckey]

    with open(f'logs/{name_prefix}logs.pickle', 'wb') as filehandle:
        pickle.dump(current_logs, filehandle)



if __name__ == '__main__':
    main(args)
