import os
import subprocess
import json
import pandas as pd

features_root_base = "C:/Users/Mahon/Documents/Research/CLAM/Phase3A_Baseline_Features/Evals"
embed_root_base = "C:/Users/Mahon/Documents/Research/CLAM/EVAL_tangle_embeddings"
labels_root_base = "C:/Users/Mahon/Documents/Research/CLAM/Labels/Boolean"
results_dir_base = "C:/Users/Mahon/Documents/Research/CMTA/results"
ckpt_path = "C:/Users/Mahon/Documents/Research/CMTA/runs/debug/best.ckpt"

summary = []

summary_rows = []

for dataset in os.listdir(features_root_base):

    dataset_results_dir = os.path.join(results_dir_base, f"{dataset}_results")
    os.makedirs(dataset_results_dir, exist_ok=True)

    cmd = [
        "python", "evaluate.py",
        "--features_root", os.path.join(features_root_base, dataset),
        "--embed_root", os.path.join(embed_root_base, dataset),
        "--labels_root", labels_root_base,
        "--features_type", "pt",
        "--ckpt", ckpt_path,
        "--results_dir", os.path.join(results_dir_base, f"{dataset}_results"),
    ]

    subprocess.run(cmd, check=True)

 # Load metrics JSON
    json_path = os.path.join(dataset_results_dir, "metrics_evaluate.json")
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    # Save individual CSV
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df.to_csv(os.path.join(dataset_results_dir, "results.csv"), index=False)

    # Append to summary
    metrics["Dataset"] = dataset
    summary_rows.append(metrics)

# Save combined summary
summary_df = pd.concat(summary)
summary_df.to_csv(os.path.join(results_dir_base, "summary_of_all_evaluations.csv"), index=False)
