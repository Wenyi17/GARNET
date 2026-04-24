"""
train_garnet.py -- Train GARNET (final: No-MP, L=1) on one dataset.

Final GARNET architecture:
  - V-free, message-passing-free evidence-gated Q/K scorer
  - Single scoring layer (L=1)
  - Three modalities:  scRNA expression (PCA) + Borzoi DNA seq + GenePT text

Usage:
  python train_garnet.py --dataset hESC
  python train_garnet.py --dataset mESC
  python train_garnet.py --dataset MouseKidney
"""

import argparse
import math
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
# NOTE: this is the EXACT model that produced the hESC=0.9418 / hHep=0.9255
# numbers in v4_L1_hh_45312838.out (params = 7,477,857).  model_v4.py is a
# cleaner V-free rewrite with 6.4M params but slightly different init, so it
# will NOT reproduce those numbers bit-for-bit.  For final reporting we stick
# with the script that generated the benchmark.
from model_no_mp import GARNETv3_NoMP


# ---------------------------------------------------------------------------
# Paths & dataset catalog
# ---------------------------------------------------------------------------
BEELINE_EXPR = "/nas/longleaf/home/leyudai/Beeline/BEELINE-data/inputs/scRNA-Seq"
BEELINE_NET  = "/nas/longleaf/home/leyudai/Beeline/Networks"
NEW_DATA_DIR = "/nas/longleaf/home/leyudai/GARNET/new_datasets"
GOLD_DIR     = f"{NEW_DATA_DIR}/gold_standards"
OUT_DIR      = "/nas/longleaf/home/leyudai/GARNET"

BORZOI_HUMAN_EMB   = "/nas/longleaf/home/leyudai/Brozoi_GRN/hESC_borzoi_gene_embed.npy"
BORZOI_HUMAN_GENES = "/nas/longleaf/home/leyudai/Brozoi_GRN/hESC_borzoi_gene_list.txt"
BORZOI_MOUSE_EMB   = "/nas/longleaf/home/leyudai/Brozoi_GRN/mHSCL_borzoi_gene_embed.npy"
BORZOI_MOUSE_GENES = "/nas/longleaf/home/leyudai/Brozoi_GRN/mHSCL_borzoi_gene_list.txt"
GENEPT_EMB_PATH    = ("/nas/longleaf/home/leyudai/GenePT/GenePT_emebdding_v2/"
                      "GenePT_gene_protein_embedding_model_3_text.pickle")

DATASETS = {
    # ---- BEELINE ------------------------------------------------------------
    "hESC":   dict(expr=f"{BEELINE_EXPR}/hESC/ExpressionData.csv",
                   gold=f"{BEELINE_NET}/human/hESC-ChIP-seq-network.csv",
                   species="human"),
    "hHep":   dict(expr=f"{BEELINE_EXPR}/hHep/ExpressionData.csv",
                   gold=f"{BEELINE_NET}/human/HepG2-ChIP-seq-network.csv",
                   species="human"),
    "mESC":   dict(expr=f"{BEELINE_EXPR}/mESC/ExpressionData.csv",
                   gold=f"{BEELINE_NET}/mouse/mESC-ChIP-seq-network.csv",
                   species="mouse"),
    "mHSC-E": dict(expr=f"{BEELINE_EXPR}/mHSC-E/ExpressionData.csv",
                   gold=f"{BEELINE_NET}/mouse/mHSC-ChIP-seq-network.csv",
                   species="mouse"),
    "mHSC-L": dict(expr=f"{BEELINE_EXPR}/mHSC-L/ExpressionData.csv",
                   gold=f"{BEELINE_NET}/mouse/mHSC-ChIP-seq-network.csv",
                   species="mouse"),
    "mDC":    dict(expr=f"{BEELINE_EXPR}/mDC/ExpressionData.csv",
                   gold=f"{BEELINE_NET}/mouse/mDC-ChIP-seq-network.csv",
                   species="mouse"),
    # ---- additional single-cell atlases ------------------------------------
    "Pancreas":    dict(expr=f"{NEW_DATA_DIR}/Pancreas/ExpressionData.csv",
                        gold=f"{GOLD_DIR}/STRING_human.csv",
                        species="human"),
    "MouseKidney": dict(expr=f"{NEW_DATA_DIR}/MouseKidney/ExpressionData.csv",
                        gold=f"{GOLD_DIR}/STRING_mouse.csv",
                        species="mouse"),
    "MouseLiver":  dict(expr=f"{NEW_DATA_DIR}/MouseLiver/ExpressionData.csv",
                        gold=f"{GOLD_DIR}/STRING_mouse.csv",
                        species="mouse"),
    "MouseHeart":  dict(expr=f"{NEW_DATA_DIR}/MouseHeart/ExpressionData.csv",
                        gold=f"{GOLD_DIR}/STRING_mouse.csv",
                        species="mouse"),
}


# ---------------------------------------------------------------------------
# Hyper-parameters (final GARNET)
# ---------------------------------------------------------------------------
SEED_DATA    = 42
N_FOLDS      = 10
D_MODEL      = 512
D_K          = 64
D_FF         = 1024       # used only by GARNETv3_NoMP signature (no FFN active)
N_HEADS      = 8
N_LAYERS     = 1          # <-- GARNET final: single scoring layer
DROPOUT      = 0.15       # unused in No-MP forward but kept for signature
PCA_DIM      = 256
LR           = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 500
WARMUP       = 30
LABEL_SMOOTH = 0.05
GENEPT_DIM   = 3072

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Shared embedding loaders
# ---------------------------------------------------------------------------
def load_shared_embeddings():
    print("Loading GenePT ...")
    with open(GENEPT_EMB_PATH, "rb") as f:
        genept_raw = pickle.load(f)
    print(f"  GenePT keys: {len(genept_raw)}")

    print("Loading Borzoi (human) ...")
    borzoi_human_genes = [g.strip() for g in open(BORZOI_HUMAN_GENES)]
    borzoi_human_emb = np.load(BORZOI_HUMAN_EMB)
    borzoi_human_map = {g: i for i, g in enumerate(borzoi_human_genes)}
    borzoi_dim_human = borzoi_human_emb.shape[1]
    print(f"  Human: {len(borzoi_human_genes)} genes, dim={borzoi_dim_human}")

    print("Loading Borzoi (mouse) ...")
    borzoi_mouse_genes = [g.strip() for g in open(BORZOI_MOUSE_GENES)]
    borzoi_mouse_emb = np.load(BORZOI_MOUSE_EMB)
    borzoi_mouse_map = {g: i for i, g in enumerate(borzoi_mouse_genes)}
    borzoi_mouse_map_upper = {g.upper(): i for i, g in enumerate(borzoi_mouse_genes)}
    borzoi_dim_mouse = borzoi_mouse_emb.shape[1]
    print(f"  Mouse: {len(borzoi_mouse_genes)} genes, dim={borzoi_dim_mouse}")

    return dict(genept_raw=genept_raw,
                borzoi_human_emb=borzoi_human_emb,
                borzoi_human_map=borzoi_human_map,
                borzoi_dim_human=borzoi_dim_human,
                borzoi_mouse_emb=borzoi_mouse_emb,
                borzoi_mouse_map=borzoi_mouse_map,
                borzoi_mouse_map_upper=borzoi_mouse_map_upper,
                borzoi_dim_mouse=borzoi_dim_mouse)


def build_borzoi_emb(universe, species, shared):
    """Return (emb matrix [n_genes, d_borzoi], #matched)."""
    if species == "human":
        emb = shared["borzoi_human_emb"]
        mp  = shared["borzoi_human_map"]
        mp_upper = {g.upper(): i for g, i in mp.items()}
        dim = shared["borzoi_dim_human"]
    else:
        emb = shared["borzoi_mouse_emb"]
        mp  = shared["borzoi_mouse_map"]
        mp_upper = shared["borzoi_mouse_map_upper"]
        dim = shared["borzoi_dim_mouse"]

    X = np.zeros((len(universe), dim), dtype=np.float32)
    matched = 0
    for i, g in enumerate(universe):
        idx = mp.get(g)
        if idx is None:
            idx = mp_upper.get(g.upper())
        if idx is not None:
            X[i] = emb[idx]
            matched += 1
    print(f"  Borzoi -> {dim}D, {matched}/{len(universe)} matched")
    return X


def build_genept_emb(universe, shared):
    """Return (emb matrix [n_genes, GENEPT_DIM], #matched).

    GenePT keys are mostly upper-case human symbols.  We look up the
    gene name both as-is and upper-cased.
    """
    raw = shared["genept_raw"]
    X = np.zeros((len(universe), GENEPT_DIM), dtype=np.float32)
    matched = 0
    for i, g in enumerate(universe):
        v = raw.get(g)
        if v is None:
            v = raw.get(g.upper())
        if v is not None:
            arr = np.asarray(v, dtype=np.float32).ravel()
            if arr.size >= GENEPT_DIM:
                X[i] = arr[:GENEPT_DIM]
            else:
                X[i, :arr.size] = arr
            matched += 1
    print(f"  GenePT -> {GENEPT_DIM}D, {matched}/{len(universe)} matched")
    return X


# ---------------------------------------------------------------------------
# Dataset loader (FULL / UNCAPPED)
# ---------------------------------------------------------------------------
def load_dataset(ds_name, cfg, shared):
    print(f"\nLoading {ds_name} (species={cfg['species']}) ...")

    expr_df = pd.read_csv(cfg["expr"], index_col=0)
    expr_genes = list(expr_df.index)
    upper2expr = {g.upper(): g for g in expr_genes}

    gold = pd.read_csv(cfg["gold"])
    gold_genes = set(gold["Gene1"]).union(set(gold["Gene2"]))

    def map_gene(g):
        if g in expr_genes:
            return g
        return upper2expr.get(g.upper())

    gold["Gene1"] = gold["Gene1"].apply(map_gene)
    gold["Gene2"] = gold["Gene2"].apply(map_gene)
    gold = gold.dropna(subset=["Gene1", "Gene2"])
    gold_in_expr = set(gold["Gene1"]).union(set(gold["Gene2"]))

    universe = [g for g in expr_genes if g in gold_in_expr] + \
               [g for g in expr_genes if g not in gold_in_expr]
    # keep ordering deterministic
    universe = sorted(set(universe), key=universe.index)

    print(f"  Gene subset: {len(universe)} (gold={len(gold_in_expr)}, hvg=0)")

    g2i = {g: i for i, g in enumerate(universe)}
    tf_set = set(gold["Gene1"])
    tf_names = [g for g in universe if g in tf_set]
    tf_idx   = np.array([g2i[g] for g in tf_names], dtype=np.int64)

    expr_mat = expr_df.loc[universe].values.astype(np.float32)
    print(f"  {len(universe)} genes, {expr_mat.shape[1]} cells, "
          f"{len(tf_names)} TFs (matched)")

    # positive TF->gene pairs (both in universe, Gene1 is TF)
    pos_set = {(g2i[r["Gene1"]], g2i[r["Gene2"]])
               for _, r in gold.iterrows()
               if r["Gene1"] in g2i and r["Gene2"] in g2i
               and r["Gene1"] in tf_set}
    pos_pairs = np.array(sorted(pos_set), dtype=np.int64)
    print(f"  Subsampled pos pairs: {len(pos_pairs)} (from {len(pos_pairs)})")

    # balanced negatives (TF, non-target) uniformly at random
    rng = np.random.RandomState(SEED_DATA)
    tf_idx_set = set(tf_idx.tolist())
    neg_pairs = []
    need = len(pos_pairs)
    pos_lookup = {(int(t), int(g)) for t, g in pos_pairs}
    while len(neg_pairs) < need:
        t = int(rng.choice(tf_idx))
        g = int(rng.randint(0, len(universe)))
        if t == g:
            continue
        if (t, g) in pos_lookup:
            continue
        neg_pairs.append((t, g))
    neg_pairs = np.array(neg_pairs, dtype=np.int64)

    pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
    labels = np.concatenate([np.ones(len(pos_pairs), dtype=np.float32),
                             np.zeros(len(neg_pairs), dtype=np.float32)])
    print(f"  {len(pos_pairs)} pos + {len(neg_pairs)} neg = {len(pairs)} pairs")

    # PCA on expression
    n_comp = min(PCA_DIM, expr_mat.shape[1] - 1, expr_mat.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED_DATA)
    X_exp = pca.fit_transform(expr_mat).astype(np.float32)
    print(f"  expression PCA -> {n_comp}D  (var={pca.explained_variance_ratio_.sum():.3f})")

    X_seq = build_borzoi_emb(universe, cfg["species"], shared)
    X_txt = build_genept_emb(universe, shared)

    return dict(genes=universe, n_genes=len(universe),
                tf_idx=tf_idx, pairs=pairs, labels=labels,
                x_exp=X_exp, x_seq=X_seq, x_txt=X_txt)


# ---------------------------------------------------------------------------
# Optim helpers
# ---------------------------------------------------------------------------
def get_cosine_warmup_scheduler(optimizer, warmup, total):
    def lr_lambda(epoch):
        progress = max(epoch, 0)
        if progress < warmup:
            return progress / max(1, warmup)
        p = (progress - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def smooth_bce(logit, target, smoothing=LABEL_SMOOTH):
    target = target * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logit, target)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_fold(data, train_pairs, train_labels, fold_idx, n_folds):
    x_exp = torch.from_numpy(data["x_exp"]).to(device)
    x_seq = torch.from_numpy(data["x_seq"]).to(device)
    x_txt = torch.from_numpy(data["x_txt"]).to(device)
    tf_idx_t = torch.from_numpy(data["tf_idx"]).to(device)
    tf_to_row = {int(t): i for i, t in enumerate(data["tf_idx"].tolist())}

    model = GARNETv3_NoMP(
        d_exp_pca=x_exp.shape[1],
        d_borzoi=x_seq.shape[1],
        d_genept=x_txt.shape[1],
        d_model=D_MODEL, d_k=D_K, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_warmup_scheduler(optimizer, WARMUP, EPOCHS)

    tr_t = torch.tensor([tf_to_row[int(t)] for t, _ in train_pairs],
                        dtype=torch.long, device=device)
    tr_g = torch.tensor([int(g) for _, g in train_pairs],
                        dtype=torch.long, device=device)
    tr_y = torch.from_numpy(train_labels).to(device)

    last_w = None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        u_agg, _A_agg, alpha_last, _all_A, w = model(
            x_exp, x_seq, x_txt, tf_idx_t)
        logits = u_agg[tr_t, tr_g]
        loss = smooth_bce(logits, tr_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_w = w.detach().cpu().numpy()

        if ep == 1 or ep % 100 == 0 or ep == EPOCHS:
            cur_lr = optimizer.param_groups[0]["lr"]
            w_str = "/".join(f"{x:.2f}" for x in last_w)
            print(f"    ep {ep:>3d}  loss={loss.item():.4f}  "
                  f"lr={cur_lr:.1e}  w=[{w_str}]")

    model.eval()
    with torch.no_grad():
        u_agg, _A_agg, alpha_last, _all_A, w = model(
            x_exp, x_seq, x_txt, tf_idx_t)
    return model, u_agg.detach().cpu().numpy(), \
           alpha_last.detach().cpu().numpy(), last_w, time.time() - t0


def eval_fold(u_mat, alpha_last, pairs, labels, data):
    tf_to_row = {int(t): i for i, t in enumerate(data["tf_idx"].tolist())}
    tr_t = np.array([tf_to_row[int(t)] for t, _ in pairs])
    tr_g = np.array([int(g) for _, g in pairs])
    scores = u_mat[tr_t, tr_g]
    pred   = (scores > 0).astype(int)

    # average alpha over heads+test TFs for reporting (alpha_last: [n_tf, n_gene, 3])
    a = alpha_last[tr_t, tr_g]          # [N, 3]
    a_mean = a.mean(axis=0)

    return dict(
        auroc      = roc_auc_score(labels, scores),
        auprc      = average_precision_score(labels, scores),
        acc        = accuracy_score(labels, pred),
        f1         = f1_score(labels, pred),
        recall     = recall_score(labels, pred),
        precision  = precision_score(labels, pred),
        alpha_bind   = float(a_mean[0]),
        alpha_coexpr = float(a_mean[1]),
        alpha_know   = float(a_mean[2]),
    )


def run_dataset(ds_name, shared):
    cfg = DATASETS[ds_name]
    data = load_dataset(ds_name, cfg, shared)

    pairs  = data["pairs"]
    labels = data["labels"]

    print(f"\n=================================================================")
    print(f"GARNET (No-MP, L={N_LAYERS})  10-fold CV on {ds_name}   device={device}")
    print(f"  d={D_MODEL} dk={D_K} heads={N_HEADS} layers={N_LAYERS}")
    print(f"  lr={LR} wd={WEIGHT_DECAY} ep={EPOCHS} warmup={WARMUP}")
    print(f"  PCA={PCA_DIM} label_smooth={LABEL_SMOOTH}")

    probe = GARNETv3_NoMP(
        d_exp_pca=data["x_exp"].shape[1],
        d_borzoi=data["x_seq"].shape[1],
        d_genept=data["x_txt"].shape[1],
        d_model=D_MODEL, d_k=D_K, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
    )
    print(f"  params: {sum(p.numel() for p in probe.parameters()):,}")
    print(f"=================================================================\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_DATA)
    rows = []

    for fi, (tr_i, te_i) in enumerate(skf.split(pairs, labels), 1):
        print(f"-- Fold {fi}/{N_FOLDS}  train={len(tr_i)} test={len(te_i)} --")
        tr_pairs, tr_labels = pairs[tr_i], labels[tr_i]
        te_pairs, te_labels = pairs[te_i], labels[te_i]

        _, u_mat, alpha_last, layer_w, secs = train_fold(
            data, tr_pairs, tr_labels, fi, N_FOLDS)

        m = eval_fold(u_mat, alpha_last, te_pairs, te_labels, data)
        print(f"   > AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
              f"ACC={m['acc']:.4f}  F1={m['f1']:.4f}")
        print(f"     alpha=[{m['alpha_bind']:.3f}/{m['alpha_coexpr']:.3f}/"
              f"{m['alpha_know']:.3f}]  "
              f"layer_w=[{'/'.join(f'{x:.2f}' for x in layer_w)}]  ({int(secs)}s)\n")

        rows.append(dict(
            fold=fi, time_s=secs,
            alpha_bind=m["alpha_bind"], alpha_coexpr=m["alpha_coexpr"],
            alpha_know=m["alpha_know"],
            layer_w="/".join(f"{x:.2f}" for x in layer_w),
            auroc=m["auroc"], auprc=m["auprc"], acc=m["acc"], f1=m["f1"],
            recall=m["recall"], precision=m["precision"],
        ))

    df = pd.DataFrame(rows)
    aur = df["auroc"]; aup = df["auprc"]; acc = df["acc"]; f1 = df["f1"]
    rec = df["recall"]; pre = df["precision"]
    print(f"\n=================================================================")
    print(f"GARNET (No-MP, L={N_LAYERS})  {ds_name}  "
          f"{N_FOLDS}-fold  MEAN +/- STD")
    print(f"  AUROC      = {aur.mean():.4f} +/- {aur.std():.4f}")
    print(f"  AUPRC      = {aup.mean():.4f} +/- {aup.std():.4f}")
    print(f"  ACC        = {acc.mean():.4f} +/- {acc.std():.4f}")
    print(f"  F1         = {f1.mean():.4f} +/- {f1.std():.4f}")
    print(f"  RECALL     = {rec.mean():.4f} +/- {rec.std():.4f}")
    print(f"  PRECISION  = {pre.mean():.4f} +/- {pre.std():.4f}")
    print(f"=================================================================")

    folds_out = f"{OUT_DIR}/results_garnet_folds_{ds_name}.tsv"
    df.to_csv(folds_out, sep="\t", index=False)
    print(f"Per-fold -> {folds_out}")

    summary_out = f"{OUT_DIR}/results_garnet_{ds_name}.tsv"
    with open(summary_out, "w") as f:
        f.write("auroc\tauprc\tacc\tf1\n")
        f.write(f"{aur.mean():.4f}+/-{aur.std():.4f}\t"
                f"{aup.mean():.4f}+/-{aup.std():.4f}\t"
                f"{acc.mean():.4f}+/-{acc.std():.4f}\t"
                f"{f1.mean():.4f}+/-{f1.std():.4f}\n")
    print(f"Summary  -> {summary_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    args = ap.parse_args()

    print("=" * 60)
    print(f"  GARNET  (No-MP, L={N_LAYERS})  --  {args.dataset}")
    print(f"  Device: {device}  |  Torch: {torch.__version__}")
    print("=" * 60)

    shared = load_shared_embeddings()
    run_dataset(args.dataset, shared)


if __name__ == "__main__":
    main()
