import torch

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


[0.25, 0.5, 0.25] * [0.25, 0.5, 0.25].T = [ 0.25 ]_3x3

