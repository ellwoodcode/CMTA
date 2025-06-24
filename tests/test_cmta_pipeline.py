import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch


def run_train(features_root, embed_root, labels_root, splits_dir, features_type, results_dir):
    cmd = [
        sys.executable,
        'train.py',
        '--features_root', features_root,
        '--embed_root', embed_root,
        '--labels_root', labels_root,
        '--splits_dir', splits_dir,
        '--features_type', features_type,
        '--batch_size', '2',
        '--num_epochs', '1',
        '--lr', '0.01',
        '--detach_align',
        '--results_dir', results_dir,
    ]
    # run in CMTA root
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    subprocess.check_call(cmd, cwd=cwd)


def create_synthetic_data(tmp_path, features_type):
    features_root = tmp_path / 'features'
    features_root.mkdir()
    embed_root = tmp_path / 'embed'
    embed_root.mkdir()
    labels_root = tmp_path / 'labels'
    labels_root.mkdir()
    splits_dir = tmp_path / 'splits'
    splits_dir.mkdir()

    slide_ids = [f'slide1.{features_type}', f'slide2.{features_type}']
    n_tiles, feat_dim = 4, 8
    # create feature files
    for sid in slide_ids:
        path = features_root / sid
        if features_type == 'pt':
            arr = torch.randn(n_tiles, feat_dim)
            torch.save(arr, path)
        else:
            with h5py.File(path, 'w') as f:
                f.create_dataset('features', data=np.random.randn(n_tiles, feat_dim))

    # create embedding files
    embed_dim = 16
    for sid in slide_ids:
        cid = sid.split('.')[0]
        arr = np.random.randn(embed_dim)
        embed_path = embed_root / f"{cid}.npy"
        np.save(str(embed_path), arr)

    # label CSV
    label_df = pd.DataFrame({'slide_id': slide_ids, 'label': [0, 1]})
    label_df.to_csv(labels_root / 'labels.csv', index=False)

    # splits CSV
    rows = []
    for i, sid in enumerate(slide_ids):
        rows.append({
            'slide_id': sid,
            'train': 1 if i == 0 else 0,
            'val': 1 if i == 1 else 0,
            'test': 0,
        })
    split_df = pd.DataFrame(rows)
    # write boolean-mask style split file (e.g. splits_0_bool.csv)
    split_df.to_csv(splits_dir / 'splits_0_bool.csv', index=False)

    return str(features_root), str(embed_root), str(labels_root), str(splits_dir)


@pytest.mark.parametrize('features_type', ['pt', 'h5'])
def test_cmta_pipeline(tmp_path, features_type):
    features_root, embed_root, labels_root, splits_dir = create_synthetic_data(tmp_path, features_type)
    results_dir = tmp_path / f'results_{features_type}'
    run_train(features_root, embed_root, labels_root, splits_dir, features_type, str(results_dir))
    metrics_file = results_dir / 'metrics.json'
    assert metrics_file.exists()
    metrics = json.load(open(metrics_file))
    assert 'val_AUROC' in metrics
    assert 'test_AUROC' in metrics