"""RealMLP (PyTorch) の学習・OOF生成スクリプト.

ベース実装: kernels/yekenot_realmlp/ps-s6-e9-realmlp-pytorch.ipynb
 (RealMLP-TD: PBLD周期埋め込み + NTP線形 + 残差ブロック + n_ens 内部アンサンブル)

目的:
  GBDT3種 (LGBM/XGB/CatBoost, OOF相関 0.993-0.995) に対して非相関な NN の OOF を作り、
  4モデルアンサンブルの利得を稼ぐ。

必須要件:
  CV は StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 固定 (他モデルと fold 共通)。

使い方:
  uv run python 04_fe_run_realmlp.py                     # 本番 (5-fold フル)
  uv run python 04_fe_run_realmlp.py --folds 1 --subsample 0.1 --epochs 1 --tag bench
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

import importlib
_fe_realmlp = importlib.import_module("03_fe_realmlp")
EXACT_TE_SMOOTHS = _fe_realmlp.EXACT_TE_SMOOTHS
ID = _fe_realmlp.ID
TARGET = _fe_realmlp.TARGET
build_features = _fe_realmlp.build_features
build_te_key_frame = _fe_realmlp.build_te_key_frame
load_orig = _fe_realmlp.load_orig
target_encode_highcard = _fe_realmlp.target_encode_highcard

warnings.filterwarnings("ignore")

# src/ からリポジトリルートを指す(data/ や oof/ はルート基準)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_PATH = os.path.join(ROOT, "data", "EV_Adoption_and_Range_Anxiety_Dataset.csv")


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing (numerical)
# ══════════════════════════════════════════════════════════════════════════════
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """median_center → robust_scale → smooth_clip (RealMLP-TD 標準の数値前処理)."""

    def __init__(self, tfms):
        self._tfms = [
            t
            for t in tfms
            if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")
        ]

    def fit(self, X: np.ndarray, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :].astype(np.float32)
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :].astype(np.float32)
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════
class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)])
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        # x: (batch, n_ens, n_cat)
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx : idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx : feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))
            features.append(torch.cat(feat_embs, dim=1))
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    """Neural Tangent Parametrization 風の線形層 (n_ens 並列)."""

    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None
        self._scale = 1.0 / math.sqrt(in_features)

    def forward(self, x):
        # x: (batch, n_ens, in) -> (n_ens, batch, in) @ (n_ens, in, out)
        # CPU では einsum より bmm の方が速いため permute + bmm を使う (数式は同一)
        out = torch.bmm(x.permute(1, 0, 2), self.weight).permute(1, 0, 2) * self._scale
        if self.bias is not None:
            out = out + self.bias
        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_ens, dim, dropout, activation=nn.SiLU):
        super().__init__()
        self.linear = NTPLinear(n_ens=n_ens, in_features=dim, out_features=dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.ones(n_ens, dim) * 0.1)

    def forward(self, x):
        residual = x
        x = self.drop(self.act(self.linear(x)))
        return residual + x * self.res_scale.unsqueeze(0)


class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay embedding (数値列の周期埋め込み)."""

    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4,
                 freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.empty(n_ens, n_features, hidden_dim))
        nn.init.normal_(self.w1, mean=0.0, std=freq_scale / math.sqrt(hidden_dim))
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(
            2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0))
        )
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0)
        )
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"],
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]
        act = cfg["activation"]

        self._dropout_modules = []
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        in_dim = total_dim
        first_linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=hidden_dims[0])
        self.first_linear = first_linear
        layers.extend([first_linear, act()])
        in_dim = hidden_dims[0]
        for hdim in hidden_dims[1:]:
            if in_dim != hdim:
                layers.extend([NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=hdim), act()])
                in_dim = hdim
            block = ResidualBlock(n_ens=n_ens, dim=hdim, dropout=cfg["dropout"], activation=act)
            self._dropout_modules.append(block.drop)
            layers.append(block)
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)
        with torch.no_grad():
            self.output_layer.weight.mul_(0.1)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        combined = torch.cat([self.num_embed(x_num), self.cate(x_cat)], dim=2)
        return self.output_layer(self.hidden(combined))  # (batch, n_ens, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Schedules / param groups / loss
# ══════════════════════════════════════════════════════════════════════════════
def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    if sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    if sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    if sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    if sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model, p):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    LR, WD = p["lr"], p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"]},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"]},
        {"params": other_w_p, "lr": LR, "weight_decay": WD},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"]},
    ]


def binary_bce_loss(y_true, logits, ls=0.0, pos_weight=None):
    if ls > 0.0:
        y_true = y_true * (1.0 - ls) + 0.5 * ls
    if pos_weight is None:
        return ((1.0 - y_true) * logits + F.softplus(-logits)).mean()
    return (
        (1.0 - y_true) * logits
        + (1.0 + (pos_weight - 1.0) * y_true) * F.softplus(-logits)
    ).mean()


# ══════════════════════════════════════════════════════════════════════════════
# sklearn-like wrapper
# ══════════════════════════════════════════════════════════════════════════════
class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, cfg):
        self.params = cfg

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None):
        p = self.params
        dev = torch.device(p["device"])
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"]).fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            cat_dims = (
                np.maximum(X_tr_cat.max(axis=0), X_val_cat.max(axis=0)) + 1
            ).tolist()
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        self.classes_ = np.unique(y_tr)
        t_build = time.time()
        self.model_ = RealMLP(
            output_dim=1, cat_dims=cat_dims, n_numerical=X_tr_num.shape[1], cfg=p
        ).to(dev)
        if verbose >= 2:
            n_par = sum(q.numel() for q in self.model_.parameters())
            print(f"  model built in {time.time() - t_build:.1f}s | params={n_par:,} | "
                  f"cat_dims(max)={max(cat_dims) if cat_dims else 0} sum={sum(cat_dims)}", flush=True)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.float32, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)
        del X_tr_num, X_tr_cat, X_val_num, X_val_cat

        n_ens = p["n_ens"]
        train_bs, eval_bs = p["train_bs"], p["eval_bs"]
        epochs, lr_sched, flat_ratio = p["epochs"], p["lr_sched"], p["flat_ratio"]
        ema_decay = p["ema_decay"]
        n_train = len(y_tr)
        total_steps = epochs * n_train
        train_order = torch.arange(n_train, device=dev)

        # EMA: state_dict のクローンを毎ステップ作ると Python オーバーヘッドが大きいため、
        # 浮動小数点パラメータへの参照を一度だけ取り、_foreach で in-place 更新する。
        live_params = [v for v in self.model_.state_dict().values() if torch.is_floating_point(v)]
        ema_buf = None
        if ema_decay > 0:
            ema_buf = [v.detach().clone() for v in live_params]

        best_score, best_epoch, best_val_probs, best_state = -np.inf, 0, None, None
        t_fold = time.time()

        log_every = p.get("log_every", 0)
        for epoch in range(epochs):
            self.model_.train()
            step = 0
            for start in range(0, n_train, train_bs):
                step += 1
                if log_every and step % log_every == 0:
                    print(f"    [ep{epoch + 1} step {step}/{math.ceil(n_train / train_bs)}] "
                          f"{time.time() - t_fold:.0f}s", flush=True)
                progress = (epoch * n_train + start) / total_steps
                idx_batch = train_order[start : start + train_bs]
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
                optimizer.zero_grad(set_to_none=True)
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = binary_bce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens), y_pred.reshape(-1),
                    ls=ls_val, pos_weight=None,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_buf is not None:
                    with torch.no_grad():
                        torch._foreach_mul_(ema_buf, ema_decay)
                        torch._foreach_add_(ema_buf, live_params, alpha=1.0 - ema_decay)

            train_order = train_order[torch.randperm(n_train, device=dev)]

            # ── Validation (EMA 重みで評価) ─────────────────────────────
            self.model_.eval()
            live_backup = None
            if ema_buf is not None:
                with torch.no_grad():
                    live_backup = [v.detach().clone() for v in live_params]
                    for pt, et in zip(live_params, ema_buf):
                        pt.copy_(et)
            with torch.no_grad():
                val_pred = np.concatenate([
                    torch.sigmoid(self.model_(Xvn[s : s + eval_bs], Xvc[s : s + eval_bs]))
                    .mean(dim=1).squeeze(-1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
            epoch_score = roc_auc_score(y_v, val_pred)
            improved = epoch_score > best_score
            if improved:
                best_score, best_epoch = epoch_score, epoch + 1
                best_val_probs = val_pred.copy()
                src = ema_buf if ema_buf is not None else live_params
                best_state = [t.detach().clone() for t in src]
            if live_backup is not None:
                with torch.no_grad():
                    for pt, lt in zip(live_params, live_backup):
                        pt.copy_(lt)
                del live_backup

            if verbose >= 2:
                print(f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                      f"best={best_score:.5f}  ls={ls_val:.4f} drop={drop_val:.4f} "
                      f"[{time.time() - t_fold:.0f}s]" + ("  *" if improved else ""), flush=True)

            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"]
                            + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch + 1} (best {best_epoch})")
                    break

        if best_state is not None:
            with torch.no_grad():
                for pt, bt in zip(live_params, best_state):
                    pt.copy_(bt)
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  -> best score: {best_score:.5f} (epoch {best_epoch})", flush=True)
        return self

    def predict_proba_pos(self, X: pd.DataFrame) -> np.ndarray:
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = np.clip(
            X[self.cat_col_names_].values.astype(np.int64), 0, np.array(self.cat_dims_) - 1
        )
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                torch.sigmoid(self.model_(Xn[s : s + eval_bs], Xc[s : s + eval_bs]))
                .mean(dim=1).squeeze(-1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG (yekenot 公開実装と同一)
# ══════════════════════════════════════════════════════════════════════════════
def make_config(**over):
    cfg = {
        # --- architecture ---
        "n_ens": 8, "embed_dim": 6, "onehot_thresh": 4,
        "hidden_dims": [256, 256, 256], "dropout": 0.05,
        "p_drop_sched": "expm4t", "activation": nn.SiLU, "add_front_scale": True,
        # --- PBLD ---
        "pbld_hidden_dim": 20, "pbld_out_dim": 5, "pbld_freq_scale": 5.0,
        "pbld_activation": nn.PReLU, "pbld_lr_factor": 0.093,
        # --- optimizer ---
        "lr": 0.01, "mom": 0.9, "sq_mom": 0.99, "lr_sched": "flat_anneal",
        "flat_ratio": 0.3, "first_layer_lr_factor": 1.2, "first_layer_wd_factor": 0.1,
        "lr_scale_mult": 10.0, "lr_bias_mult": 0.1, "weight_decay": 0.013,
        "wd_scale_mult": 0.1, "wd_bias_mult": 0.5,
        "ema_decay": 0.997875, "grad_clip": 1.0,
        # --- label smoothing ---
        "ls_eps": 0.04, "ls_eps_sched": "cos",
        # --- preprocessing ---
        "tfms": ["median_center", "robust_scale", "smooth_clip"],
        # --- training ---
        "epochs": 2, "train_bs": 256, "eval_bs": 8192, "verbosity": 2,
        "use_early_stopping": False,
        "early_stopping_additive_patience": 10,
        "early_stopping_multiplicative_patience": 1,
        # --- device ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_state": 42,
    }
    cfg.update(over)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="実際に回す fold 数 (分割は常に5)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-ens", type=int, default=8)
    ap.add_argument("--train-bs", type=int, default=256)
    ap.add_argument("--subsample", type=float, default=1.0, help="train を割合でサブサンプル")
    ap.add_argument("--threads", type=int, default=0, help="torch スレッド数 (0=自動)")
    ap.add_argument("--log-every", type=int, default=0, help="N step ごとに進捗を出力")
    ap.add_argument("--tag", type=str, default="realmlp", help="出力ファイル名の接尾辞")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--exact-te", action="store_true",
        help="厳密値TE(高カーデ2列+Smooth Keys3本, Triple smooth=auto/10/100 の15列)を追加",
    )
    ap.add_argument(
        "--no-orig", action="store_true",
        help="元データ(EV_Adoption...csv)由来の org_mean 特徴量を使わない。外部データの寄与の切り分け用",
    )
    ap.add_argument(
        "--digits", action="store_true",
        help="高カーデ2列(年収・通勤距離)の各桁をカテゴリ特徴として追加",
    )
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    seed_everything(42)

    cfg = make_config(epochs=args.epochs, n_ens=args.n_ens, train_bs=args.train_bs,
                      log_every=args.log_every)
    print(f"PyTorch {torch.__version__} | device={cfg['device']} | "
          f"threads={torch.get_num_threads()} | cuda={torch.cuda.is_available()}", flush=True)

    # ── Load ──────────────────────────────────────────────────────────────
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    orig = None if args.no_orig else load_orig(ORIG_PATH)
    print(f"orig data: {'loaded ' + str(orig.shape) if orig is not None else 'NOT FOUND (org_mean skip)'}")

    train[TARGET] = (train[TARGET] == "Yes").astype(int)
    X = train.drop([ID, TARGET], axis=1)
    y = train[TARGET]
    train_id, test_id = train[ID], test[ID]
    X_test = test.drop([ID], axis=1)
    del train, test

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    category_map = {}
    t0 = time.time()
    X, new_cat_cols, new_num_cols, combo_names = build_features(
        X, cat_cols, num_cols, category_map, fit=True, orig=orig, digits=args.digits
    )
    X_test, _, _, _ = build_features(
        X_test, cat_cols, num_cols, category_map, fit=False, orig=orig, digits=args.digits
    )
    cat_cols = cat_cols + new_cat_cols
    num_cols = num_cols + new_num_cols
    print(f"FE done in {time.time() - t0:.0f}s | cat={len(cat_cols)} num={len(num_cols)} "
          f"| X={X.shape} X_test={X_test.shape}", flush=True)

    # ── 厳密値TE用キーフレーム(高カーデ2列+Smooth Keys, 教師なしなので全体で作ってよい)──
    te_keys_all = te_keys_test = None
    if args.exact_te:
        te_keys_all = build_te_key_frame(X)
        te_keys_test = build_te_key_frame(X_test)
        print(f"exact-te keys: {list(te_keys_all.columns)} -> "
              f"{len(te_keys_all.columns) * len(EXACT_TE_SMOOTHS)} 列追加予定", flush=True)

    # ── CV (全モデル共通・変更禁止) ────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    n_run = 0
    t_all = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if fold > args.folds:
            break
        if args.subsample < 1.0:
            rng = np.random.RandomState(42)
            tr_idx = rng.choice(tr_idx, int(len(tr_idx) * args.subsample), replace=False)
            val_idx = rng.choice(val_idx, int(len(val_idx) * args.subsample), replace=False)

        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tst = X_test.copy()

        # ── fold 内 Target Encoding (リーク防止: fit は X_tr のみ) ─────────
        te = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=42)
        tr_enc = te.fit_transform(X_tr[combo_names], y_tr)
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names] = tr_enc
        X_val[te_names] = te.transform(X_val[combo_names])
        X_tst[te_names] = te.transform(X_tst[combo_names])

        # ── 厳密値TE(高カーデ2列+Smooth Keys, 入れ子CVでリーク防止) ──────
        if args.exact_te:
            key_cols = list(te_keys_all.columns)
            ht_tr, (ht_val, ht_tst) = target_encode_highcard(
                te_keys_all.iloc[tr_idx],
                y_tr.to_numpy(),
                [te_keys_all.iloc[val_idx], te_keys_test],
                key_cols,
                smooths=EXACT_TE_SMOOTHS,
                n_inner=5,
                seed=42,
            )
            X_tr[list(ht_tr.columns)] = ht_tr
            X_val[list(ht_val.columns)] = ht_val
            X_tst[list(ht_tst.columns)] = ht_tst

        if fold == 1:
            print(f"len(FEATURES): {X_tr.shape[1]}", flush=True)
        print(f"{'#' * 16}\n### Fold {fold}/5  (train={len(y_tr)}, val={len(y_val)})\n{'#' * 16}", flush=True)

        seed_everything(42 + fold)
        t_fold = time.time()
        model = RealMLP_TD_Classifier(cfg)
        model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols)

        val_preds = model.best_val_probs_
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba_pos(X_tst) / args.folds

        fs = roc_auc_score(y_val, val_preds)
        fold_scores.append(fs)
        n_run += 1
        print(f"Fold {fold} | AUC {fs:.5f} | {time.time() - t_fold:.0f}s\n", flush=True)
        del X_tr, X_val, X_tst, model

    print("=" * 40)
    print(f"Fold AUCs : {[f'{s:.5f}' for s in fold_scores]}")
    print(f"Fold mean : {np.mean(fold_scores):.5f}")
    if n_run == 5 and args.subsample >= 1.0:
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"\033[1mOverall OOF AUC: {oof_auc:.5f}\033[0m")
    print(f"total {time.time() - t_all:.0f}s")
    print("=" * 40, flush=True)

    if args.no_save or n_run < 5 or args.subsample < 1.0:
        print("(部分実行のため成果物は保存しません)")
        return

    tag = args.tag
    os.makedirs(os.path.join(ROOT, "oof"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "submit"), exist_ok=True)
    np.save(os.path.join(ROOT, "oof", f"oof_{tag}.npy"), oof_preds)
    np.save(os.path.join(ROOT, "oof", f"pred_{tag}.npy"), test_preds)
    pd.DataFrame({ID: test_id, TARGET: test_preds}).to_csv(
        os.path.join(ROOT, "submit", f"submission_{tag}.csv"), index=False
    )
    print(f"saved: oof/oof_{tag}.npy, oof/pred_{tag}.npy, submit/submission_{tag}.csv")

    # ── 他モデルとの相関 (アンサンブル価値の確認) ─────────────────────────
    from scipy.stats import spearmanr

    for other in ["lgbm", "xgb", "catboost"]:
        path = os.path.join(ROOT, "oof", f"oof_{other}.npy")
        if os.path.exists(path):
            o = np.load(path)
            print(f"  spearman(realmlp, {other:8s}) = {spearmanr(oof_preds, o).statistic:.4f}"
                  f"   [{other} OOF AUC {roc_auc_score(y, o):.5f}]")


if __name__ == "__main__":
    main()
