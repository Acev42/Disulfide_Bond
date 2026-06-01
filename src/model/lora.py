import torch
import torch.nn.functional as F
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
        # x batch,in
        # A in,r
        # B r,out

class LinearWithLoRA(nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B # Combine LoRA matrices
        # Then combine LoRA with orig. weights
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        return F.linear(x, combined_weight, self.linear.bias)

class LinearWithDoRAMerged(nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        self.lora_m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))
  # Code loosely inspired by
  # https://github.com/catid/dora/blob/main/dora.py
    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        numerator = self.linear.weight + self.lora.alpha*lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator
        new_weight = self.lora_m * directional_component
        return F.linear(x, new_weight, self.linear.bias)

def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)

def add_lora(model, rank, alpha):
    for i, layer in enumerate(model.layers):
        layer.self_attn.k_proj = LinearWithLoRAMerged(layer.self_attn.k_proj, rank, alpha)
        layer.self_attn.v_proj = LinearWithLoRAMerged(layer.self_attn.v_proj, rank, alpha)
        layer.self_attn.q_proj = LinearWithLoRAMerged(layer.self_attn.q_proj, rank, alpha)
        layer.fc1 = LinearWithLoRAMerged(layer.fc1, rank, alpha)
        layer.fc2 = LinearWithLoRAMerged(layer.fc2, rank, alpha)
    for name, param in model.named_parameters():
        if name.count("lora") == 0:
            param.requires_grad = False
    return model

def add_dora(model, rank, alpha):
    for i, layer in enumerate(model.layers):
        layer.self_attn.v_proj = LinearWithDoRAMerged(layer.self_attn.v_proj, rank, alpha)
        layer.self_attn.q_proj = LinearWithDoRAMerged(layer.self_attn.q_proj, rank, alpha)
    for name, param in model.named_parameters():
        if name.count("lora") == 0:
            param.requires_grad = False
    return model
