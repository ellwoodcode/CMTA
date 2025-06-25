#!/usr/bin/env python
"""
Evaluate a trained CMTABinary model on the entire cohort (all slides).
"""

import argparse
import json
import logging
import os
import sys

# ensure imports from CMTA and sibling CLAM regardless of CWD
_ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_ROOT)
sys.path.append(os.path.abspath(os.path.join(_ROOT, '..', 'CLAM')))

import numpy as np
import torch
import glob
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

import h5py

from cmta_models.cmta_binary import CMTABinary

def mil_collate_fn(batch):
    """
    Collate function for MIL dataset: variable-length slide features per sample.
    Returns list of tensors (pathomics), stacked embeddings, tensor labels, list of slide_ids.
    """
    x_paths, x_embs, ys, sids = zip(*batch)
    x_embs = torch.stack(x_embs)
    ys = torch.tensor(ys)
    return list(x_paths), x_embs, ys, list(sids)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x_paths, x_embs, y_batch, _ in loader:
            for x_path, x_embed, y in zip(x_paths, x_embs, y_batch):
                # each slide in its own batch
                if x_path.dim() == 3:
                    xp = x_path.unsqueeze(0).to(device)
                else:
                    xp = x_path.to(device)
                xe = x_embed.unsqueeze(0).to(device)
                y = y.float().to(device)
                logits, _, _, _ = model(xp, xe)
                ys.append(y.item())
                ps.append(logits.item())
    ys = np.array(ys)
    ps = np.array(ps)
    try:
        auc = roc_auc_score(ys, ps)
    except ValueError:
        auc = float('nan')
    ap = average_precision_score(ys, ps)
    acc = ((ps > 0) == ys).mean()
    return auc, ap, acc


def evaluate_cohort(
    features_root,
    embed_root,
    labels_root,
    label_name,
    features_type,
    ckpt,
    fusion="bilinear",
    model_size="small",
    path_size=None,
    batch_size=8,
    results_dir=None,
):
    """
    Programmatically evaluate CMTABinary on a cohort specified by label_name.

    Args:
        features_root (str): root dir of slide features (pt_files/h5_files).
        embed_root (str): root dir of embedding .npy files.
        labels_root (str): root dir of Boolean label CSVs.
        label_name (str): basename (without .csv) of the label file to load.
        features_type (str): 'pt' or 'h5'.
        ckpt (str): path to trained checkpoint.
        fusion (str): CMTA fusion type.
        model_size (str): CMTA model size.
        path_size (int): override pathomics input dim.
        batch_size (int): DataLoader batch size.
        results_dir (str or None): if set, save metrics JSON here.
    Returns:
        dict: AUROC, AUPRC, acc.
    """
    # load label CSV
    label_file = os.path.join(labels_root, f"{label_name}.csv")
    if not os.path.isfile(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    label_df = pd.read_csv(label_file)
    if 'slide_id' not in label_df.columns or 'label' not in label_df.columns:
        raise ValueError("Label CSV must contain slide_id and label columns")

    # build dataset
    class EvalDataset(Dataset):
        def __init__(self, df, froot, eroot, ftype):
            self.df = df.reset_index(drop=True)
            self.froot = froot
            self.eroot = eroot
            self.ftype = ftype
            self._h5 = {}

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            sid = self.df.loc[idx, 'slide_id']
            lbl = int(self.df.loc[idx, 'label'])
            if self.ftype == 'pt':
                feat = torch.load(os.path.join(self.froot, 'pt_files', f"{sid}.pt"))
            else:
                if sid not in self._h5:
                    self._h5[sid] = h5py.File(
                        os.path.join(self.froot, 'h5_files', f"{sid}.h5"), 'r'
                    )
                feat = torch.from_numpy(self._h5[sid]['features'][:])
            cid = os.path.splitext(sid)[0]
            emb = torch.from_numpy(np.load(os.path.join(self.eroot, f"{sid}.npy")))
            return feat, emb, lbl, sid

    ds = EvalDataset(label_df, features_root, embed_root, features_type)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    sample = ds[0]
    in_dim = sample[0].shape[-1]
    emb_dim = sample[1].shape[-1]

    model = CMTABinary(
        omic_sizes=[emb_dim],
        fusion=fusion,
        model_size=model_size,
        path_size=path_size or in_dim,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load checkpoint, allowing missing or unexpected keys for backward compatibility
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=mil_collate_fn)
    scores = evaluate(model, loader, device)
    results = {'AUROC': scores[0], 'AUPRC': scores[1], 'acc': scores[2]}
    if results_dir:
        out_file = os.path.join(results_dir, f"{label_name}_metrics.json")
        with open(out_file, 'w') as f:
            json.dump(results, f)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CMTABinary on the full cohort"
    )
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--embed_root", required=True)
    parser.add_argument("--labels_root", required=True)
    parser.add_argument(
        "--features_type", choices=["pt", "h5"], required=True
    )
    parser.add_argument(
        "--ckpt", required=True, help="Path to trained best.ckpt"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--fusion", choices=["concat", "bilinear"], default="concat"
    )
    parser.add_argument(
        "--model_size", choices=["small", "large"], default="small"
    )
    parser.add_argument("--results_dir", required=True)
    return parser.parse_args()


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(results_dir, 'evaluate.log')),
        ],
    )


def main():
    args = parse_args()
    setup_logging(args.results_dir)
    logging.info(f"Args: {args}")

    # load full label CSV (e.g. labels_ims1.csv)
    embed_base = os.path.basename(args.embed_root).lower().split('_')[0]
    print(embed_base)
    if embed_base == "fondaz":
        embed_base = "Lombardy-Italy"
    if embed_base == "milan-fondaz":
        embed_base = "Milan-Italy"
    if embed_base == "carolina":
        embed_base = "N.Carolina"
    pattern = f"*{embed_base}*.csv"
    label_files = glob.glob(os.path.join(args.labels_root, pattern))
    if not label_files:
        raise ValueError(f"No label CSV found in {args.labels_root}")
    label_df = pd.read_csv(label_files[0])
    if 'slide_id' not in label_df.columns or 'label' not in label_df.columns:
        raise ValueError("Label CSV must contain slide_id and label columns")
    logging.info(f"Loaded {len(label_df)} samples from {label_files[0]}")

    # build evaluation dataset over all slides
    class EvalDataset(Dataset):
        def __init__(self, df, froot, eroot, ftype):
            self.df = df.reset_index(drop=True)
            self.froot = froot
            self.eroot = eroot
            self.ftype = ftype
            self._h5 = {}

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            sid = self.df.loc[idx, 'slide_id']
            lbl = int(self.df.loc[idx, 'label'])
            if self.ftype == 'pt':
                fpath = os.path.join(self.froot, 'pt_files', f"{sid}.pt")
                feat = torch.load(fpath)
            else:
                fpath = os.path.join(self.froot, 'h5_files', f"{sid}.h5")
                if sid not in self._h5:
                    self._h5[sid] = h5py.File(fpath, 'r')
                feat = torch.from_numpy(self._h5[sid]['features'][:])
            cid = os.path.splitext(sid)[0]
            emb_path = os.path.join(self.eroot, f"{sid}.npy")
            emb = torch.from_numpy(np.load(emb_path))
            return feat, emb, lbl, sid

    ds = EvalDataset(label_df, args.features_root, args.embed_root, args.features_type)
    logging.info(f"Eval dataset size: {len(ds)}")

    # infer dims and build model
    path_feat, embed_feat, _, _ = ds[0]
    in_path_dim = path_feat.shape[-1]
    embed_dim = embed_feat.shape[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CMTABinary(
        omic_sizes=[embed_dim], fusion=args.fusion,
        model_size=args.model_size, path_size=in_path_dim
    )
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=mil_collate_fn)
    auc, ap, acc = evaluate(model, loader, device)
    results = {'AUROC': auc, 'AUPRC': ap, 'acc': acc}
    logging.info(f"Evaluation over full cohort: {results}")

    out_file = os.path.join(args.results_dir, 'metrics_evaluate.json')
    with open(out_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Saved metrics to {out_file}")


if __name__ == '__main__':
    main()