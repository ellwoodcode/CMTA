"""
Example command for training CMTA binary task:

python train.py \
    --features_root /path/to/Phase3A_Baseline_Features \
    --embed_root /path/to/IMS1_tangle_embeddings \
    --labels_root /path/to/Labels/Boolean \
    --splits_dir /path/to/splits/task_1_tumor_vs_normal_100_ims1_tangle \
    --features_type pt \
    --batch_size 8 \
    --num_epochs 20 \
    --lr 1e-4 \
    --detach_align \
    --results_dir runs/tumor_vs_normal
"""