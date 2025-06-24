#!/usr/bin/env python
"""
Evaluate a trained CMTABinary model on validation and test splits.
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
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from datasets.wsi_embed_dataset import WSIEmbedDataset
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CMTABinary on val/test splits")
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--embed_root", required=True)
    parser.add_argument("--labels_root", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--features_type", choices=["pt", "h5"], required=True)
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fusion", default="concat", choices=["concat", "bilinear"] )
    parser.add_argument("--model_size", default="small", choices=["small", "large"])
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

    # pick boolean-mask split CSV
    bool_files = sorted(f for f in os.listdir(args.splits_dir) if f.endswith('_bool.csv'))
    if not bool_files:
        raise ValueError(f"No boolean-mask CSV in {args.splits_dir}")
    splits_file = os.path.join(args.splits_dir, bool_files[0])
    logging.info(f"Using splits file: {splits_file}")

    # build datasets
    val_ds = WSIEmbedDataset(
        args.features_root, args.embed_root, args.labels_root,
        splits_file, 'val', args.features_type
    )
    test_ds = WSIEmbedDataset(
        args.features_root, args.embed_root, args.labels_root,
        splits_file, 'test', args.features_type
    )
    logging.info(f"Eval dataset sizes val={len(val_ds)}, test={len(test_ds)}")

    # infer dims and build model
    sample = val_ds[0]
    in_path_dim = sample[0].shape[-1]
    embed_dim = sample[1].shape[-1]
    model = CMTABinary(
        omic_sizes=[embed_dim], fusion=args.fusion,
        model_size=args.model_size, path_size=in_path_dim
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    # loaders
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=mil_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=mil_collate_fn)

    # run evaluation
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    results = {
        'val_AUROC': val_metrics[0], 'val_AUPRC': val_metrics[1], 'val_acc': val_metrics[2],
        'test_AUROC': test_metrics[0], 'test_AUPRC': test_metrics[1], 'test_acc': test_metrics[2],
    }
    logging.info(f"Evaluation results: {results}")

    # save to JSON
    out_file = os.path.join(args.results_dir, 'metrics_evaluate.json')
    with open(out_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Saved metrics to {out_file}")


if __name__ == '__main__':
    main()