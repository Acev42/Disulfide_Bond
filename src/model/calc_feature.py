# input = 主链原子坐标 、 序列字符串
# output = 节点特征（未拼接）：相对坐标系 、 序列embedding、 边特征
import torch
import torch.nn.functional as F
def nan_to_num(tensor, nan=0.0):
    tensor[torch.isnan(tensor)] = nan
    return tensor


# 把nan值替换为
def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


# 归一化，将tensor最外层维度中的每一个元素都除以第二范数，简单来说就是把张量视作一个列表，对列表内每一个子张量分别进行归一化

def _dihedrals( coo, consistancy, eps=1e-7):
    '''
    from https://github.com/EricZhangSCUT/SPIN-CGNN/blob/main/code/features.py
    coo [L, 5, 3]
    consistancy [L, 2]
    D_features: sin/cos of 3 dihedral and 3 angle with consistancy [L,6]+[L,6]+[L,2]
    '''
    backbone = coo[:, :3]
    backbone = backbone.flatten(0, 1)  # [L*3, 3]

    dX = backbone[1:, :] - backbone[:-1, :]  # [L*3-1, 3]
    U = _normalize(dX, dim=-1)  # [L*3-1, 3]
    u_0 = U[:-2, :]  # [L*3-3, 3]
    u_1 = U[1:-1, :]  # [L*3-3, 3]
    u_2 = U[2:, :]  # [L*3-3, 3]

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)  # [L*3-3, 3]
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)  # [L*3-3, 3]

    cosD = (n_0 * n_1).sum(-1)  # [L*3-3]
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)  # [L*3-3]
    v = _normalize(torch.cross(n_0, n_1), dim=-1)  # [L*3-3, 3]
    D = torch.sign((-v * u_1).sum(-1)) * torch.acos(cosD)  # [L*3-3]
    D = F.pad(D, (1, 2), 'constant', 0)  # [L*3]
    D = D.view(-1, 3)  # [L, 3]
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), -1)  # [L, 6]

    # alpha, beta, gamma
    cosD = (u_0 * u_1).sum(-1)  # [L*3-3]
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)  # [L*3-3]
    D = torch.acos(cosD)  # [L*3-3]
    D = F.pad(D, (1, 2), 'constant', 0)  # [L*3]
    D = D.view(-1, 3)  # [L, 3]
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), -1)  # [L, 6]

    consistancy = torch.cat([
        consistancy, consistancy[:, 1].unsqueeze(1),  # [L, 3] cos
        consistancy, consistancy[:, 1].unsqueeze(1),  # [L, 3] sin
    ], dim=-1)  # [L, 6]
    D_features = torch.cat([
        Dihedral_Angle_features * consistancy,  # [L, 6]
        Angle_features * consistancy,  # [L, 6]
        consistancy[:, :2]  # [L, 2]
    ], dim=-1)
    return D_features  # [L, 14]