"""
Dataset returning (img_tensor, embed_tensor, label, slide_id).
Image tensor = features loaded from .pt or .h5; embed_tensor = numpy.load of matching .npy
(same filename stem, searched under --embed_root). Case_id is slide_id.split('.')[0].
Cache opened h5 handles.
"""

import glob
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WSIEmbedDataset(Dataset):
    """
    Dataset for loading slide features and TANGLE embeddings with train/val/test splits.

    Args:
        features_root (str): directory with slide feature files (.pt or .h5).
        embed_root (str): directory with embedding .npy files.
        labels_root (str): directory with boolean label CSVs.
        splits_file (str): CSV with boolean mask columns and optional label column.
        split (str): one of 'train', 'val', or 'test'.
        features_type (str): 'pt' or 'h5', specifies which slide features to load.
    """

    def __init__(
        self,
        features_root,
        embed_root,
        labels_root,
        splits_file,
        split,
        features_type,
    ):
        self.features_root = features_root
        self.embed_root = embed_root
        self.labels_root = labels_root
        self.splits_file = splits_file
        self.split = split
        self.features_type = features_type
        self._h5_cache = {}
        df = pd.read_csv(self.splits_file)
        if 'slide_id' not in df.columns:
            df.rename(columns={df.columns[0]: 'slide_id'}, inplace=True)
        if "label" not in df.columns:
            embed_base = os.path.basename(self.embed_root).lower().split("_")[0]
            pattern = f"*{embed_base}*.csv"
            label_files = glob.glob(os.path.join(self.labels_root, pattern))
            if not label_files:
                label_files = glob.glob(os.path.join(self.labels_root, "*.csv"))
            if not label_files:
                raise ValueError(f"No label CSV found in {self.labels_root}")
            label_df = pd.read_csv(label_files[0])
            if "slide_id" not in label_df.columns or "label" not in label_df.columns:
                raise ValueError("Label CSV must contain slide_id and label columns")
            df = df.merge(label_df[["slide_id", "label"]], on="slide_id", how="left")

        mask = df.get(self.split)
        if mask is None:
            raise ValueError(
                f"Split column '{self.split}' not found in {self.splits_file}"
            )
        df = df[mask == 1].reset_index(drop=True)
        self.slide_ids = df["slide_id"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        # load slide-level features from .pt or .h5, appending extension
        if self.features_type == "pt":
            feat_fname = f"pt_files/{slide_id}.pt"
        elif self.features_type == "h5":
            feat_fname = f"h5_files/{slide_id}.h5"
        else:
            raise ValueError(f"Unsupported features_type: {self.features_type}")
        feat_path = os.path.join(self.features_root, feat_fname)
        if self.features_type == "pt":
            features = torch.load(feat_path)
        else:
            # HDF5: cache file handles for speed
            if slide_id not in self._h5_cache:
                self._h5_cache[slide_id] = h5py.File(feat_path, "r")
            h5_file = self._h5_cache[slide_id]
            arr = h5_file["features"][:]
            features = torch.from_numpy(arr)

        embed_path = os.path.join(self.embed_root, f"{slide_id}.npy")
        embed_arr = np.load(embed_path)
        embed_tensor = torch.from_numpy(embed_arr)

        return features, embed_tensor, label, slide_id