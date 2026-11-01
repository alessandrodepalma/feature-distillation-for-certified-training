import torch
import sys
import os
import json
import time
import gc
import copy

from torch import nn
from torch.nn.parameter import Parameter

from tools.custom_torch_modules import Add, Mul, Flatten
import tools.bab_tools.vnnlib_utils as vnnlib_utils
from tools.bab_tools.model_utils import one_vs_all_from_model as ovalbab_1vsall_constructor, add_single_prop, \
    simplify_network, reluified_max_pool
import tools.bab_tools.bab_runner as ovalbab_runner
from plnn.naive_approximation import NaiveNetwork
from models.utils import Flatten as shi_flatten, LatentHelper

# Disable OVAL BaB's verbose printing.
class do_not_print:
    # Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def convert_layers(x, layers):

    def reshaper(inp, ndim):
        if inp.dim() < ndim:
            return inp.view(inp.shape + (1,) * (ndim - 1))
        else:
            return inp

    converted_layers = []
    if isinstance(layers, LatentHelper):
        layers = layers.layers
    for lay in layers:
        if isinstance(lay, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            assert lay.track_running_stats

            ndim = x.dim() - 1
            mean = reshaper(lay.running_mean, ndim)
            var = reshaper(lay.running_var, ndim)
            weight = reshaper(lay.weight, ndim)
            bias = reshaper(lay.bias, ndim)

            op1 = Add(-mean)
            op2 = Mul(1/torch.sqrt(var + lay.eps))
            op3 = Mul(weight)
            op4 = Add(bias)
            x = torch.nn.Sequential(op1, op2, op3, op4)(x)
            converted_layers.extend([op1, op2, op3, op4])

        elif isinstance(lay, shi_flatten):
            x = lay(x)
            converted_layers.append(Flatten())
        else:
            x = lay(x)
            converted_layers.append(lay)

    return torch.nn.Sequential(*converted_layers)


def get_ovalbab_network(in_example, layers):

    model = convert_layers(in_example.cpu(), layers)
    # check that the two models coincide in forward pass
    assert (model(in_example.cpu()) - layers(in_example.cpu())).abs().max() < 1e-3

    # Assert that the model specification is currently supported.
    supported = vnnlib_utils.is_supported_model(model)
    assert supported

    layers = list(model.children())
    for clayer in layers:
        if isinstance(clayer, torch.nn.Linear):
            clayer.bias.data = clayer.bias.data.squeeze(0)
    return layers


def one_vs_all_from_model_from_lbs(model, lbs, C, domain=None):
    """
        Given a pre-trained PyTorch network given by model, the true_label (ground truth) and the input domain for the
        property, create a network encoding a 1 vs. all adversarial verification task, but only for logit differences
        for which we do not have a robustness proof via auto_lirpa (lbs are the bounds, C is the logit diff matrix).
        The one-vs-all property is encoded exploiting a max-pool layer.
    """

    for p in model.parameters():
        p.requires_grad = False
    layers = list(model.children())

    last_layer = layers[-1]
    diff_in = last_layer.out_features

    assert lbs.shape[0] == 1
    diff_out = (lbs <= 0).sum()
    diff_layer = nn.Linear(diff_in, diff_out, bias=True)

    assert C.shape[0] == 1
    if diff_out > 1:
        weight_diff = -C.squeeze(0)[lbs.squeeze(0) <= 0].cpu()
    else:
        # no sign flipping is needed as there is no max pool addition
        weight_diff = C.squeeze(0)[lbs.squeeze(0) <= 0].cpu()
    bias_diff = torch.zeros(diff_out)

    diff_layer.weight = Parameter(weight_diff, requires_grad=False)
    diff_layer.bias = Parameter(bias_diff, requires_grad=False)
    layers.append(diff_layer)
    layers = simplify_network(layers)

    if diff_out > 1:
        verif_layers = [copy.deepcopy(lay).cuda() for lay in layers]
        intermediate_net = NaiveNetwork(verif_layers)
        verif_domain = domain.cuda().unsqueeze(0)
        intermediate_net.define_linear_approximation(verif_domain, override_numerical_errors=True)
        lbs = intermediate_net.lower_bounds[-1].squeeze(0).cpu()

        candi_tot = diff_out
        # since what we are actually interested in is the minium of gt-cls,
        # we revert all the signs of the last layer
        max_pool_layers = reluified_max_pool(candi_tot, lbs, flip_out_sign=True)

        # simplify linear layers
        simp_required_layers = layers[-1:] + max_pool_layers
        simplified_layers = simplify_network(simp_required_layers)

        final_layers = layers[:-1] + simplified_layers
    else:
        # there is no maxpool to add, just a plain diff yielding a scalar output
        final_layers = layers

    return final_layers


def create_1_vs_all_verification_problem(
        model, y, input_bounds, max_solver_batch, inputs, c, use_lbs=False, lbs=None, crown_1vsall=False):

    num_tolerance = 1e-4 if not crown_1vsall else 1e-1  # we use CROWN on less numerically precise conversions

    if not use_lbs:
        with do_not_print():
            verif_layers = ovalbab_1vsall_constructor(
                torch.nn.Sequential(*model), y, domain=input_bounds, max_solver_batch=max_solver_batch,
                use_ib=(not crown_1vsall), num_classes=model[-1].weight.shape[0])

        # Assert the functional equivalence of 1_vs_all with the original network
        out_diff_min = torch.nn.Sequential(*verif_layers)(inputs)
        out = torch.nn.Sequential(*model)(inputs)
        out_diff = torch.bmm(c.cpu(), out.unsqueeze(-1)).squeeze(-1)
        assert (out_diff_min - out_diff.min()).abs() < num_tolerance
    else:
        # use LBs from an autolirpa call on the same problem to eliminate unnecessary logit differences when doing
        # one vs all
        with do_not_print():
            verif_layers = one_vs_all_from_model_from_lbs(
                torch.nn.Sequential(*model), lbs, c, domain=input_bounds)

        # Assert the functional equivalence of 1_vs_all with the clipped output network
        out_diff_min = torch.nn.Sequential(*verif_layers)(inputs)
        out = torch.nn.Sequential(*model)(inputs)
        out_diff = torch.bmm(c.squeeze(0)[lbs.squeeze(0) <= 0].unsqueeze(0).cpu(), out.unsqueeze(-1)).squeeze(-1)
        assert (out_diff_min - out_diff.min()).abs() < num_tolerance

    return verif_layers


def run_oval_bab(verif_layers, input_bounds, ovalbab_json_config, timeout=20, results_table=None, test_idx=None,
                 json_name=None, record_name=None):
    # Run OVAL-BaB with the configuration specified in ovalbab_json_config
    return_dict = dict()
    start_time = time.time()

    with open(ovalbab_json_config) as json_file:
        json_params = json.load(json_file)
    with do_not_print():
        ovalbab_runner.bab_from_json(
            json_params, verif_layers, input_bounds, return_dict, None, instance_timeout=timeout, start_time=start_time)
    del json_params

    bab_out, bab_nb_states = ovalbab_runner.bab_output_from_return_dict(return_dict)
    bab_time = time.time() - start_time

    # Store BaB results in a table for later analysis.
    if results_table is not None:
        results_table.loc[test_idx]["prop"] = test_idx
        results_table.loc[test_idx][f"BSAT_{json_name}"] = bab_out
        results_table.loc[test_idx][f"BBran_{json_name}"] = bab_nb_states
        results_table.loc[test_idx][f"BTime_{json_name}"] = bab_time
        results_table.to_pickle(record_name)

    torch.cuda.empty_cache()
    gc.collect()

    return bab_out == "False", bab_out != "True"


def create_1_vs_1_verification_problem(model, y, y_other, inputs, num_class):
    verif_layers = add_single_prop(model, y, y_other, num_classes=num_class)

    # Assert the functional equivalence of 1_vs_1 with the original network
    out_diff = torch.nn.Sequential(*verif_layers)(inputs)
    out = torch.nn.Sequential(*model)(inputs).squeeze(0)
    assert (out_diff - (out[y] - out[y_other])).abs() < 1e-4
    return verif_layers

