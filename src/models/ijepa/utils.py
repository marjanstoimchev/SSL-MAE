import torch


def repeat_interleave_batch(x, B, repeat):
    """Repeat each batch element 'repeat' times (from I-JEPA)."""
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x
