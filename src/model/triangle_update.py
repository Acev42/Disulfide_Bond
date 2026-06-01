from triangle_multiplicative_module import TriangleMultiplicativeModule
from torch import nn
class EdgeTriangleUpdate(nn.Module):
    def __init__(self, edge_dim, mode="outgoing"):
        super().__init__()
        assert mode in ["outgoing", "ingoing"]
        self.edge_dim = edge_dim
        self.model = TriangleMultiplicativeModule(
            dim=edge_dim,  # feature map dimension
            hidden_dim=edge_dim * 2,  # intermediate dimension size
            mix=mode  # either 'ingoing' or 'outgoing'
        )

    def forward(self, edge, edge_index, nnode):
        # edge: Nedge, dim
        # edge_index: 2, Nedge
        edge_dim = self.edge_dim
        edge_2d = torch.zeros((nnode, nnode, edge_dim), dtype=edge.dtype, device=edge.device)
        edge_index_1d = edge_index[0] * nnode + edge_index[1]
        mask = torch.zeros(nnode * nnode, dtype=torch.bool, device=edge.device)
        mask = mask.index_fill_(0, edge_index_1d, True).view(nnode, nnode)
        o = edge_2d.view(-1, edge_dim).index_copy(0, edge_index_1d, edge)
        o = o.view(nnode, nnode, edge_dim)
        o = self.model(torch.unsqueeze(o, 0), mask=torch.unsqueeze(mask, 0))
        o = torch.squeeze(o)
        edge_new = o.view(-1, edge_dim)[edge_index_1d]
        return edge_new


class TriEdgeAttenUpdate3(nn.Module):
    def __init__(self, edge_dim, hidden_dim, nhead):
        super().__init__()
        self.edge_dim = edge_dim
        self.to_q = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(edge_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, edge_dim)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        assert hidden_dim % nhead == 0
        dim_head = int(hidden_dim / nhead)
        self.scale = dim_head ** -0.5
        self.norm_1 = nn.LayerNorm(edge_dim)
        self.norm_2 = nn.LayerNorm(edge_dim)
        self.linear = MLP(edge_dim, edge_dim * 2, 0.0)

    def forward(self, edge, triedge_index):
        # triedge_index: ntri_edge, 3
        h = self.nhead
        q = self.to_q(edge)
        k, v = self.to_kv(edge).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> n h d', h=h), (q, k, v))
        qq = torch.index_select(q, 0, triedge_index[:, 0])
        kk = torch.index_select(k, 0, triedge_index[:, 1])
        vv = torch.index_select(v, 0, triedge_index[:, 2])
        scaled_dot_prod = (qq * kk).sum(-1, keepdim=True) * self.scale  # n h 1
        atten = softmax(scaled_dot_prod, index=triedge_index[:, 0])  # n h 1
        edge_new = scatter(vv * atten, index=triedge_index[:, 0], reduce='sum')  # n h d
        edge_new = rearrange(edge_new, 'n h d -> n (h d)', h=h)
        edge_new = self.to_out(edge_new)

        # node_update = self.norm_node(self.w0(node_update) + node_feats)
        edge_new = self.norm_1(edge_new + edge)
        edge_new = self.norm_2(self.linear(edge_new) + edge_new)
        return edge_new


class TriEdgeAttenUpdate(nn.Module):
    def __init__(self, edge_dim, hidden_dim, nhead):
        super().__init__()
        self.edge_dim = edge_dim
        self.to_q = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(edge_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, edge_dim)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        assert hidden_dim % nhead == 0
        dim_head = int(hidden_dim / nhead)
        self.scale = dim_head ** -0.5
        self.norm_1 = nn.LayerNorm(edge_dim)
        self.norm_2 = nn.LayerNorm(edge_dim)

    def forward(self, edge, triedge_index):
        h = self.nhead
        q = self.to_q(edge)
        k, v = self.to_kv(edge).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> n h d', h=h), (q, k, v))
        qq = torch.index_select(q, 0, triedge_index[:, 0])
        kk = torch.index_select(k, 0, triedge_index[:, 1])
        vv = torch.index_select(v, 0, triedge_index[:, 1])
        scaled_dot_prod = (qq * kk).sum(-1, keepdim=True) * self.scale  # n h 1
        atten = softmax(scaled_dot_prod, index=triedge_index[:, 0])  # n h 1
        edge_new = scatter(vv * atten, index=triedge_index[:, 0], reduce='sum')  # n h d
        edge_new = rearrange(edge_new, 'n h d -> n (h d)', h=h)
        edge_new = self.to_out(edge_new)

        edge_new = self.norm_1(edge_new + edge)
        return edge_new

