from torch import distributed as dist

def synchronize():
    """Helper function to synchronize (barrier)
        among all processes when using distributed training"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()

def get_rank():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    return rank
