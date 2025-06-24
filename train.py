"""
Train CMTA on a binary slide-level task using slide features and TANGLE embeddings.
"""

import argparse
import json
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_ROOT)
sys.path.append(os.path.abspath(os.path.join(_ROOT, '..', 'CLAM')))

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score)
from torch.utils.data import DataLoader

def mil_collate_fn(batch):
    """
    Collate function for MIL dataset: returns list of variable-length feature tensors,
    stacked embeddings, tensor labels, and slide ids.
    """
    x_paths, x_embs, ys, sids = zip(*batch)
    x_embs = torch.stack(x_embs)
    ys = torch.tensor(ys)
    return list(x_paths), x_embs, ys, list(sids)

from datasets.wsi_embed_dataset import WSIEmbedDataset
from cmta_models.cmta_binary import CMTABinary


def parse_args():
    parser = argparse.ArgumentParser(description="Train CMTA binary task")
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--embed_root", required=True)
    parser.add_argument("--labels_root", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--features_type", choices=["pt", "h5"], required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--detach_align", action="store_true", default=True,
        help="Detach p and g for L1 alignment loss"
    )
    parser.add_argument(
        "--no_detach_align", dest="detach_align", action="store_false",
        help="Do not detach p and g for L1 alignment loss"
    )
    parser.add_argument("--results_dir", required=True)
    return parser.parse_args()


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(os.path.join(results_dir, 'train.log'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x_paths, x_embs, y_batch, _ in loader:
            for x_path, x_embed, y in zip(x_paths, x_embs, y_batch):
                x_path = x_path.unsqueeze(0).to(device)
                x_embed = x_embed.unsqueeze(0).to(device)
                y = y.float().to(device)
                logits, _, _, _ = model(x_path, x_embed)
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


def main():
    args = parse_args()
    logger = setup_logging(args.results_dir)
    logger.info(f"Args: {args}")

    bool_files = sorted(
        f for f in os.listdir(args.splits_dir) if f.endswith('_bool.csv')
    )
    if not bool_files:
        raise ValueError(f"No boolean-mask split CSV (*_bool.csv) in {args.splits_dir}")
    splits_file = os.path.join(args.splits_dir, bool_files[0])
    logger.info(f"Using boolean-mask splits file: {splits_file}")

    train_ds = WSIEmbedDataset(
        args.features_root, args.embed_root, args.labels_root,
        splits_file, 'train', args.features_type
    )
    val_ds = WSIEmbedDataset(
        args.features_root, args.embed_root, args.labels_root,
        splits_file, 'val', args.features_type
    )
    test_ds = WSIEmbedDataset(
        args.features_root, args.embed_root, args.labels_root,
        splits_file, 'test', args.features_type
    )
    logger.info(
        f"Dataset sizes train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    sample = train_ds[0]
    path_feat, embed_feat, _, _ = sample
    in_path_dim = path_feat.shape[-1]
    embed_dim = embed_feat.shape[-1]
    model = CMTABinary(
        omic_sizes=[embed_dim],
        detach_align=args.detach_align,
        path_size=in_path_dim,
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=mil_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=mil_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=mil_collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = 0.0
    metrics = {
        'val_AUROC': [], 'test_AUROC': [],
        'val_AUPRC': [], 'test_AUPRC': [],
        'val_acc': [], 'test_acc': [],
    }

    device = next(model.parameters()).device
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_paths, x_embs, y_batch, _ in train_loader:
            losses = []
            for x_path, x_embed, y in zip(x_paths, x_embs, y_batch):
                x_path = x_path.unsqueeze(0).to(device)
                x_embed = x_embed.unsqueeze(0).to(device)
                y = y.float().to(device)
                logits, align_loss, _, _ = model(x_path, x_embed)
                losses.append(model.loss(logits, y, align_loss))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        val_auc, val_pr, val_acc = evaluate(model, val_loader, device)
        test_auc, test_pr, test_acc = evaluate(model, test_loader, device)
        metrics['val_AUROC'].append(val_auc)
        metrics['val_AUPRC'].append(val_pr)
        metrics['val_acc'].append(val_acc)
        metrics['test_AUROC'].append(test_auc)
        metrics['test_AUPRC'].append(test_pr)
        metrics['test_acc'].append(test_acc)

        logger.info(
            f"Epoch {epoch}/{args.num_epochs}, "
            f"loss={avg_loss:.4f}, val_AUROC={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                model.state_dict(), os.path.join(args.results_dir, 'best.ckpt')
            )

    with open(os.path.join(args.results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()