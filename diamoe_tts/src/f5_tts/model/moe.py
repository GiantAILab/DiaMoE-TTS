import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

class TransformerExpert(nn.Module):
    def __init__(self, input_dim=512, num_layers=1, num_heads=8, dropout=0.1, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 2 * input_dim

        self.input_dim = input_dim
        self.pos_embed = None  # Initial state is empty, and it is dynamically constructed during runtime.

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_sinusoidal_embedding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        return pe

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.size()
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, but got {D}"

        # Dynamic construction or expansion of positional encoding
        if self.pos_embed is None or self.pos_embed.size(1) < T:
            self.pos_embed = self._build_sinusoidal_embedding(T, D).to(x.device)

        pos = self.pos_embed[:, :T, :]
        pos = pos.to(dtype=x.dtype)
        x = x + pos
        return self.encoder(x)  # [B, T, D]



class MLPExpert(nn.Module):
    def __init__(
        self,
        input_dim: int=512,        # text_embed:(batch,length, 512)
        hidden_dims: list = None,             # e.g., [512, 256]
        output_dim: int = None,
        activation: str = "GELU"       # e.g., "ReLU", "GELU", "LeakyReLU", "Tanh"
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if hidden_dims is None:
            hidden_dims = [2*input_dim]

        activation_dict = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "ELU": nn.ELU
        }

        assert activation in activation_dict, f"Unsupported activation: {activation}"

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_dict[activation]())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        B, T, D = x.shape
        x = x.view(-1, D)  # [B*T, D]
        x = self.net(x)  # [B*T, D]
        return x.view(B, T, -1)  # [B, T, D]

class SimpleGate(nn.Module):
    def __init__(self, num_experts: int, input_dim: int=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        return: [batch_size, num_experts] logits
        """
        return self.linear(x)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        return: [batch_size, num_experts] logits
        """
        return self.linear(x)

class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        num_experts: int,
        num_experts_per_tok: int=1,
        use_residual: bool = True,
        use_dialect_clf=False,
        dialect_clf_lambda=0,
        dialect_kinds=0,

    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_residual = use_residual



        self.use_dialect_clf = use_dialect_clf
        if use_dialect_clf:
            self.dialect_clf_lambda = dialect_clf_lambda
            # self.dialect_classifier = nn.Linear(input_dim, dialect_kinds)

    def forward(self, inputs_raw: torch.Tensor, dialect_labels=None, text_embed_for_gate=None):
        # inputs_raw: [batch_size, seq_len, hidden_dim]
        B, T, D = inputs_raw.shape

        # Average pool the representation of each sample, and mask out refer during inference.
        if text_embed_for_gate is None:
            pooled_inputs = inputs_raw.mean(dim=1)  # shape: [B, D]
        else:
            pooled_inputs = text_embed_for_gate.mean(dim=1)

        gate_logits = self.gate(pooled_inputs)  # [B, num_experts]
        if self.use_dialect_clf and dialect_labels is not None:
            dialect_loss = 0.0
            # dialect_logits = self.dialect_classifier(pooled_inputs)
            dialect_labels = torch.tensor(dialect_labels, dtype=torch.long).to(gate_logits.device)

            dialect_loss = F.cross_entropy(gate_logits, dialect_labels)


        # For each sample, select the top-k experts (typically set to 1)
        topk_weights, topk_indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=1)  # [B, K]
        topk_weights = F.softmax(topk_weights, dim=1, dtype=torch.float).to(inputs_raw.dtype)
        print("Selected experts per sample:", topk_indices.squeeze().tolist())
        # init output
        results = torch.zeros_like(inputs_raw)

        for i, expert in enumerate(self.experts):
            # Find out which samples have selected the current expert
            sample_idx, kth = torch.where(topk_indices == i)  # shape: [?,]
            if sample_idx.numel() == 0:
                continue

            # Obtain all the tokens of the selected sample（[?, T, D]）
            expert_input = inputs_raw[sample_idx]  # [N, T, D]

            # feed in expert
            expert_output = expert(expert_input)

            # Weighted and accumulated to the corresponding position
            for j, idx in enumerate(sample_idx):
                results[idx] += topk_weights[idx, kth[j]] * expert_output[j]

        if self.use_residual:
            results = results + inputs_raw  # Sample-by-sample residuals

        if self.use_dialect_clf and dialect_labels is not None:
            return results, gate_logits, dialect_loss * self.dialect_clf_lambda
        else:
            return results, gate_logits

class TokenMoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        num_experts: int,
        num_experts_per_tok: int=2,
        use_residual: bool = True,
        use_dialect_clf=False,
        dialect_clf_lambda=0,
        dialect_kinds=0
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_residual = use_residual

        if use_residual:
            input_dim = experts[0].net[0].in_features  # Suppose that all experts have the same structure


        self.use_dialect_clf = use_dialect_clf
        if use_dialect_clf:
            self.dialect_clf_lambda = dialect_clf_lambda
            self.dialect_classifier = nn.Linear(input_dim, dialect_kinds)

    def forward(self, inputs_raw: torch.Tensor, dialect_labels=None):
        ishape = inputs_raw.shape
        inputs = inputs_raw.view(-1, ishape[-1])
        gate_logits = self.gate(inputs)

        # Load balancing loss
        # gates = F.softmax(gate_logits, dim=1)
        # indices1_s = torch.argmax(gates, dim=1)
        # num_experts = int(gates.shape[1])
        # mask1 = F.one_hot(indices1_s, num_classes=num_experts)
        # me = torch.mean(gates, dim=0)
        # ce = torch.mean(mask1.float(), dim=0)
        # l_aux = torch.mean(me * ce) * num_experts * num_experts

        # dialect classfication
        if self.use_dialect_clf:
            dialect_loss = 0.0
            dialect_logits = self.dialect_classifier(inputs)
            dialect_labels = torch.tensor(dialect_labels, dtype=torch.long)
            dialect_labels_expanded = dialect_labels.view(-1).repeat_interleave(ishape[1]).to(dialect_logits.device)
            dialect_loss = F.cross_entropy(dialect_logits, dialect_labels_expanded)


        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if batch_idx.numel() > 0:
                expert_out = expert(inputs[batch_idx])
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_out
        if self.use_residual:

            results = results + inputs
        results_out = results.view(ishape)
        if self.use_dialect_clf:
            return results_out, gate_logits, dialect_loss*self.dialect_clf_lambda
        else:
            return results_out, gate_logits

EXPERT_DICT = {'mlp': MLPExpert,
               'transformer': TransformerExpert}

if __name__ == "__main__":

    expert = TransformerExpert(input_dim=512, num_layers=1, num_heads=8)

    # [batch_size, seq_len, hidden_dim]
    B, T, D = 4, 10, 512
    x = torch.randn(B, T, D)

    with torch.no_grad():
        output = expert(x)
    import pdb
    pdb.set_trace()
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)















