
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def is_fsdp_model(pl_module):
    return isinstance(pl_module, FSDP) or any(isinstance(m, FSDP) for m in pl_module.modules())


def get_layers(model, layer_path):
    try:
        attributes = layer_path.split(".")
        layers = model
        for attr in attributes:
            layers = getattr(layers, attr)
        return layers
    except Exception as e:
        print(e, flush=True)
        return None


def compute_fsdp_global_mean_std(module):
    # Get local weight data once
    local_weight = module.weight.data
    local_numel = local_weight.numel()
    device = local_weight.device

    # Prepare tensors for reduction
    stats = torch.tensor([local_weight.sum(), local_numel], device=device)

    # Single all_reduce call for sum and count
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_sum, total_numel = stats

    # Compute global mean
    global_mean = global_sum / total_numel

    # Compute local squared difference from global mean (no need for .clone())
    local_sq_diff_sum = ((local_weight - global_mean) ** 2).sum()

    # Prepare for second reduction
    dist.all_reduce(local_sq_diff_sum, op=dist.ReduceOp.SUM)
    global_sq_diff_sum = local_sq_diff_sum

    # Compute global std with Bessel's correction
    # https://pytorch.org/docs/stable/generated/torch.std.html
    global_std = torch.sqrt(global_sq_diff_sum / max(0, total_numel - 1))

    return global_mean, global_std