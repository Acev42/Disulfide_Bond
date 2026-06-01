import torch
from torch_geometric.utils import softmax, scatter
from torch import nn, einsum
from einops import rearrange

class AttentionCOO(nn.Module):
    def __init__(self, dim, hidden_dim, nhead, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        assert hidden_dim % nhead == 0
        dim_head = int(hidden_dim / nhead)
        self.scale = dim_head ** -0.5
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        dim_linear_block = dim * 2

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value, coo_index):
        assert coo_index.shape[1] == 2
        h = self.nhead
        q = self.to_q(query)
        k = self.to_k(key)
        v = (self.to_v(value))
        q, k, v = map(lambda t:rearrange(t, 'n (h d) -> n h d', h=h), (q, k, v))
        qq = torch.index_select(q, 0, coo_index[:, 0])
        kk = torch.index_select(k, 0, coo_index[:, 1])
        vv = torch.index_select(v, 0, coo_index[:, 1])
        scaled_dot_prod = (qq * kk).sum(-1, keepdim=True) * self.scale  # n h 1
        atten = softmax(scaled_dot_prod, index=coo_index[:, 0])  # n h 1
        out = scatter(vv * atten, index=coo_index[:, 0], reduce='sum')  # n h d
        out = rearrange(out, 'n h d -> n (h d)', h=h)
        out = self.to_out(out)

        y = self.norm_1(self.drop(out) + query)
        out = self.norm_2(self.linear(y) + y)
        return out


def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

class FastAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.dropout = dropout
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None,
    ):
        h = self.heads
        context = default(context, x)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal, dropout_p=self.dropout)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = FastAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=0.)
        self.attn2 = FastAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=0.)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x, queries, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask = mask) #query updated
        out     = self.attn2(x, induced) #x updated

        x =  out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries

class CombineCoorAtten(nn.Module):
    def __init__(self, natom, hidden_atom):
        super().__init__()
        self.l1 = nn.Linear(3, hidden_atom, bias=False)
        self.l2 = nn.Linear(3, hidden_atom, bias=False)
        self.l3 = nn.Linear(hidden_atom, 3, bias=False)
    def forward(self, coor1, coor2):
        #coor1, coor2: Nedge nsidechain_atom 3
        coor_diff = coor1[:, None, :, :] - coor2[:, :, None, :] #nedge nsidechain, nsidechain ,3
        coor_dist = torch.linalg.vector_norm(coor_diff, dim=-1, keepdim=False) #nedge nsidechain, nsidechain
        attention = torch.softmax(-coor_dist, dim=-1)
        q = self.l2(coor1)
        v = self.l1(coor2)
        out = q + torch.einsum('... i j , ... j d -> ... i d', attention, v)
        coor = self.l3(out)
        return coor

