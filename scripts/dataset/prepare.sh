#!/bin/bash
# Download and tokenize the dataset in a pipeline to minimise disk usage.
# Each file is downloaded, tokenized, and deleted before the next one starts.
#
# Usage:
#   bash scripts/dataset/prepare.sh

source devconfig.sh

mkdir -p "$SSD_MOUNT" "$NFS_MOUNT" "$TOKENIZED_DIR"

# ── Build task queue ──────────────────────────────────────────────────────────
echo "Listing dataset files..."
base_url=https://data.commoncrawl.org/contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/

# Get file list from HuggingFace Hub (no AWS credentials needed)
python - <<EOF
from huggingface_hub import list_repo_tree
import os

repo_id = "mlfoundations/dclm-baseline-1.0"
shard_path = "global-shard_03_of_10/local-shard_1_of_10"
nfs = os.environ["NFS_MOUNT"]
ssd = os.environ["SSD_MOUNT"]

count = 0
for item in list_repo_tree(repo_id, repo_type="dataset", path_in_repo=shard_path):
    if not item.path.endswith(".jsonl.zst"):
        continue
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{item.path}"
    name = os.path.basename(item.path)
    file = os.path.join(ssd, name)
    task = os.path.join(nfs, name + ".task")
    if not os.path.exists(task):
        with open(task, "w") as f:
            f.write(url + "\n")
            f.write(file + "\n")
        count += 1

print(f"Task queue ready: {count} files queued.")
EOF

# ── Dispatch parallel workers (one per GPU) ───────────────────────────────────
NUM_GPUS=4
WORKERS_PER_GPU=$(( $(nproc) / NUM_GPUS ))

echo "Starting $NUM_GPUS parallel workers..."
for i in $(seq 0 $(( NUM_GPUS - 1 ))); do
    CUDA_VISIBLE_DEVICES=$i NWORKERS=$WORKERS_PER_GPU \
        bash scripts/dataset/modules/download_and_tokenize.sh &
done

wait
echo "All done. Tokenized files are in $TOKENIZED_DIR"
