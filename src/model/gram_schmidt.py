import torch

def rigidFrom3Points_torch(x1, x2, x3,xps=1e-5):
    # Gram_Schmidt
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2,dim=-1)
    R = torch.stack((e1, e2, e3), dim = 1)
    return R
