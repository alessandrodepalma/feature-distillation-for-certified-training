import torch
import torch.nn as nn
from utils import *
from auto_LiRPA import BoundedTensor, BoundDataParallel
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
import pdb
import math


def get_crown_loss(lb, reduce=True):
    lb_padded = torch.cat([torch.zeros_like(lb[:, :1]), lb], dim=1)
    fake_labels = torch.zeros(lb.size(0), dtype=torch.long, device=lb.device)
    if not reduce:
        robust_loss_ = ce_loss_noreduce(-lb_padded, fake_labels)
    else:
        robust_loss_ = ce_loss(-lb_padded, fake_labels)
    return robust_loss_


def get_C(args, data, labels):
    return get_spec_matrix(data, labels, args.num_class)


def get_bound(args, model, x, data=None, labels=None, meter=None, bounding_algorithm="IBP"):
    assert bounding_algorithm in ['IBP', 'CROWN-IBP']

    c, bound_lower, bound_upper = get_C(args, data, labels), True, False

    if bounding_algorithm == "IBP":
        lb, ub = model(x=(x,), method_opt="compute_bounds", IBP=True, C=c, method=None, no_replicas=True)
    else:
        # CROWN-IBP bound computation
        lb, ub = model.compute_bounds(
            x=(x,), IBP=True, C=c, method='backward', bound_lower=bound_lower, bound_upper=bound_upper)

    if meter is not None:
        update_relu_stat(model, meter)
    return lb


def ub_robust_loss(args, model, x, data, labels, meter=None, bounding_algorithm="IBP"):

    lb = get_bound(
        args, model, x, data=data, labels=labels, meter=meter, bounding_algorithm=bounding_algorithm)
    robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
    # Pad zero at the beginning for each example, and use fake label '0' for all examples
    robust_loss = get_crown_loss(lb)

    if robust_loss is not None and torch.isnan(robust_loss):
        robust_err = 1.

    return robust_loss, robust_err, lb


def fetch_distillation_features_bounds(teacher_model, student_model, distill_point, earlier_latent=False,
                                       fetch_teacher=True):
    # NOTE: assumes teacher_model is a torch model in eval mode, student_model is an autolirpa model with bounds
    # already computed
    # computes (student latent LB, student latent UB), teacher latent computed on student adversary
    # NOTE: assumes the latent space is defined as whatever is fed to the network output node
    with torch.no_grad():
        if fetch_teacher:
            if earlier_latent:
                # NOTE: earlier_latent is hardcoded for CNN7
                teacher_f = torch.nn.Sequential(*(teacher_model.layers[:-4]))
                teacher_latent = teacher_f(distill_point)
            else:
                _ = teacher_model(distill_point)
                teacher_latent = teacher_model.latent
        else:
            teacher_latent = None
    if not earlier_latent:
        last_bounds = student_model[student_model.output_name[0]].inputs[0]
    else:
        # NOTE: earlier_latent is hardcoded for CNN7
        lirpa_modules = student_model._modules.values()
        last_bounds = list(lirpa_modules)[-4].inputs[0]
    return (last_bounds.lower, last_bounds.upper), teacher_latent


def get_latent_from_lirpa(model, earlier_latent=False):
    # NOTE: assumes the latent space is whatever is fed to the final network operation
    if not earlier_latent:
        last_bounds = model[model.output_name[0]].inputs[0]
    else:
        # NOTE: earlier_latent is hardcoded for CNN7
        lirpa_modules = model._modules.values()
        last_bounds = list(lirpa_modules)[-4].inputs[0]
    return last_bounds.forward_value