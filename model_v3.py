"""
GARNET v3 — Multi-Head EFRA with Multi-Layer GRN Scoring.

Changes from v2:
  - Learnable layer-wise aggregation: final u = Σ w_l * u_l
    (weighted sum across all EFRA layers, w = softmax(learnable)).
  - Wider default config (d_model=512, 8 heads).
  - Pre-LN (LayerNorm before attention) for deeper stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadEFRALayer(nn.Module):

    def __init__(self, d_model, d_k, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_model // n_heads

        self.W_Q_seq = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K_seq = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_Q_exp = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K_exp = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_Q_txt = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K_txt = nn.Linear(d_model, n_heads * d_k, bias=False)

        self.gate_weight = nn.Parameter(torch.zeros(n_heads, 3, 3))
        self.gate_bias   = nn.Parameter(torch.zeros(n_heads, 3))
        for h in range(n_heads):
            nn.init.eye_(self.gate_weight[h])

        self.W_V_G  = nn.Linear(d_model, d_model, bias=False)
        self.W_V_TF = nn.Linear(d_model, d_model, bias=False)
        self.W_O_TF = nn.Linear(d_model, d_model, bias=False)
        self.W_O_G  = nn.Linear(d_model, d_model, bias=False)

        # Pre-LN style
        self.ln_attn_tf = nn.LayerNorm(d_model)
        self.ln_attn_g  = nn.LayerNorm(d_model)
        self.ln_ff_tf   = nn.LayerNorm(d_model)
        self.ln_ff_g    = nn.LayerNorm(d_model)

        self.ffn_tf = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.ffn_g = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

        self.attn_drop = nn.Dropout(dropout)

    def _evidence_gate(self, s_bind, s_coexpr, s_know):
        s = torch.stack([s_bind, s_coexpr, s_know], dim=-1)
        out = torch.einsum("htge,hef->htgf", s, self.gate_weight) + \
              self.gate_bias[:, None, None, :]
        return F.softmax(out, dim=-1)

    def forward(self, H_TF, H_G, z_exp, z_seq, z_txt, tf_idx):
        n_tf, n_gene = H_TF.size(0), H_G.size(0)
        H, dk, dv = self.n_heads, self.d_k, self.d_v

        # Pre-LN on hidden states for V
        H_TF_n = self.ln_attn_tf(H_TF)
        H_G_n  = self.ln_attn_g(H_G)

        Q_seq = self.W_Q_seq(z_seq[tf_idx]).view(n_tf, H, dk)
        K_seq = self.W_K_seq(z_seq).view(n_gene, H, dk)
        Q_exp = self.W_Q_exp(z_exp[tf_idx]).view(n_tf, H, dk)
        K_exp = self.W_K_exp(z_exp).view(n_gene, H, dk)
        Q_txt = self.W_Q_txt(z_txt[tf_idx]).view(n_tf, H, dk)
        K_txt = self.W_K_txt(z_txt).view(n_gene, H, dk)

        scale = math.sqrt(dk)
        s_bind   = torch.einsum("thd,ghd->htg", Q_seq, K_seq) / scale
        s_coexpr = torch.einsum("thd,ghd->htg", Q_exp, K_exp) / scale
        s_know   = torch.einsum("thd,ghd->htg", Q_txt, K_txt) / scale

        alpha = self._evidence_gate(s_bind, s_coexpr, s_know)
        u = (alpha[..., 0] * s_bind +
             alpha[..., 1] * s_coexpr +
             alpha[..., 2] * s_know)

        A = self.attn_drop(F.softmax(u, dim=-1))

        V_G  = self.W_V_G(H_G_n).view(n_gene, H, dv)
        V_TF = self.W_V_TF(H_TF_n).view(n_tf, H, dv)

        out_tf = torch.bmm(A, V_G.permute(1, 0, 2))
        out_g  = torch.bmm(A.transpose(-2, -1), V_TF.permute(1, 0, 2))

        out_tf = self.W_O_TF(out_tf.permute(1, 0, 2).reshape(n_tf, -1))
        out_g  = self.W_O_G(out_g.permute(1, 0, 2).reshape(n_gene, -1))

        H_TF = H_TF + out_tf
        H_G  = H_G  + out_g

        H_TF = H_TF + self.ffn_tf(self.ln_ff_tf(H_TF))
        H_G  = H_G  + self.ffn_g(self.ln_ff_g(H_G))

        u_mean     = u.mean(dim=0)
        A_mean     = A.mean(dim=0)
        alpha_mean = alpha.mean(dim=0)
        return H_TF, H_G, A_mean, u_mean, alpha_mean


class GARNETv3(nn.Module):
    """GARNET v3 with multi-layer scoring aggregation."""

    def __init__(self, d_exp_pca, d_borzoi, d_genept,
                 d_model=512, d_k=64, d_ff=1024,
                 n_heads=8, n_layers=4, dropout=0.15):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers

        self.enc_exp = nn.Sequential(
            nn.Linear(d_exp_pca, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.enc_seq = nn.Sequential(
            nn.Linear(d_borzoi, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.enc_txt = nn.Sequential(
            nn.Linear(d_genept, d_model), nn.LayerNorm(d_model), nn.GELU())

        self.f_uni = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            MultiHeadEFRALayer(d_model, d_k, d_ff, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Learnable layer weights for GRN aggregation
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x_exp, x_seq, x_txt, tf_idx):
        z_exp = self.enc_exp(x_exp)
        z_seq = self.enc_seq(x_seq)
        z_txt = self.enc_txt(x_txt)

        h = self.f_uni(torch.cat([z_exp, z_seq, z_txt], dim=-1))
        H_TF = h[tf_idx]
        H_G  = h

        all_A, all_u, all_alpha = [], [], []
        for layer in self.layers:
            H_TF, H_G, A, u, alpha = layer(
                H_TF, H_G, z_exp, z_seq, z_txt, tf_idx)
            all_A.append(A)
            all_u.append(u)
            all_alpha.append(alpha)

        # Weighted layer aggregation
        w = F.softmax(self.layer_logits, dim=0)
        u_agg = sum(w[l] * all_u[l] for l in range(self.n_layers))
        A_agg = sum(w[l] * all_A[l] for l in range(self.n_layers))
        alpha_last = all_alpha[-1]

        return u_agg, A_agg, alpha_last, all_A, w
