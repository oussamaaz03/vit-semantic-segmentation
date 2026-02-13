import torch
from timm import scheduler
from timm import optim

from segm.optim.scheduler import PolynomialLR


def create_scheduler(opt_args, optimizer):
    if opt_args.sched == "polynomial":
        lr_scheduler = PolynomialLR(
            optimizer,
            opt_args.poly_step_size,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_lr,
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler


def create_optimizer(opt_args, model):
    """
    Create optimizer - supports custom AdamW for Vision Transformers
    
    For AdamW: Uses optimized hyperparameters for ViT training
    For others: Falls back to timm library
    """
    if opt_args.opt.lower() == "adamw":
        # Custom AdamW with hyperparameters optimized for Vision Transformers
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=opt_args.weight_decay,
        )
        return optimizer
    else:
        # Use timm for other optimizers (sgd, adam, etc.)
        return optim.create_optimizer(opt_args, model)
