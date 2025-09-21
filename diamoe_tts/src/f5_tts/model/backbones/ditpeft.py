"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import pdb

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlocklora,
    AdaLayerNorm_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
     ConvNeXtV2BlockWithAdapter,
)


# Text embedding
class LoRAinput(nn.Module):
    """
    add LoRA
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 16, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()  # dropout for lora
        self.scaling = self.alpha / self.rank  # zoom factor

        # originally frozen weights and biases
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        # in LoRALinear.__init__

        self.weight.requires_grad = False
        self.bias.requires_grad = False
        

        # LoRA  A & B
        # A: [rank, in_features], B: [out_features, rank]
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        # init lora
        nn.init.normal_(self.lora_A, mean=0, std=1)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        original_output = F.linear(x, self.weight, self.bias)
        
        #  (x @ A^T) @ B^T
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        # dropout and scale
        lora_output = self.dropout(lora_output) * self.scaling

        return original_output + lora_output



# noised input audio and context mixing embedding
    # Text embedding
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

            # use adaptor in TextEmbedding
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)

            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2BlockWithAdapter(
                    text_dim, 
                    text_dim * conv_mult,
                    use_adapter=True,
                    adapter_compression=0.25,  # γ=0.25
                    adapter_kernel_size=3      # kernel size=3
                ) for _ in range(conv_layers)]
            )

        
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text



    
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, 
                 use_prompt_adapter=True,
                 drop_path_rate=0.3,
                 lora_rank=16):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
        for param in self.conv_pos_embed.parameters():
            param.requires_grad = False
        
        self.use_prompt_adapter = use_prompt_adapter
        self.drop_path_rate=drop_path_rate
        if use_prompt_adapter:
            # add Prompt Adapter
            #  add LoRA in proj
            self.proj = LoRAinput(self.proj, rank=lora_rank, alpha=1, dropout=0.05)
            
            # DropPath regularization
            #self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        else:
            self.drop_path = nn.Identity()
    
    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))        

        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks

class DiT_peft(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
        moe=None
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        #for param in self.text_embed.parameters():
        #    param.requires_grad = False

        #for param in self.input_embed.parameters():
        #    param.requires_grad = False
        for param in self.time_embed.parameters():
            param.requires_grad = False

        # MoE
        if moe:
            self.use_moe = True
            self.moe = moe
            for param in self.moe.parameters():
                param.requires_grad=False
        else:
            self.use_moe = False

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlocklora(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None



    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
        cache=False,
        return_features = False,
        dialect_labels=None,
        text_for_gate=None


    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)


        # MOE
        if self.use_moe:
            # The dialect classification loss is calculated only when the input is a dialect label and the usage of the dialect is lost.
            if self.moe.use_dialect_clf and dialect_labels is not None:
                if not drop_text:
                    text_embed, _, dialect_loss = self.moe(text_embed,dialect_labels)
                else:
                    dialect_loss = torch.tensor(0.0).to(text_embed.device)
            else:
                if not drop_text:
                    if text_for_gate is not None:
                        text_embed_for_gate = self.text_embed(text_for_gate, seq_len, drop_text=False)
                        text_embed, _ = self.moe(text_embed, dialect_labels, text_embed_for_gate)
                    else:
                        text_embed, _ = self.moe(text_embed, dialect_labels)


        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x
        if return_features:
            output_features = [x]
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
            if return_features:
                output_features.append(x)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        if return_features:
            if self.moe.use_dialect_clf:
                return output_features, output, dialect_loss
            else:
                return output_features, output
        else:
            if self.use_moe and self.moe.use_dialect_clf and dialect_labels is not None:
                return output, dialect_loss
            else:
                return output

