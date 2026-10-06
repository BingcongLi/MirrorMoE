# Local scratch dirs — will be created and cleaned up automatically
export SSD_MOUNT=/local/home/bingli/datasets/dclm28b/textfiles
export NFS_MOUNT=/local/home/bingli/datasets/dclm28b/tasks
export DATASET=dclm28b

# GCS bucket path where the dataset lives / will be uploaded
export GCP_DATASET_DIR=gs://your-bucket/datasets

# Which dataset to process
export DATASET=dclm28b

# HuggingFace tokenizer (for tokenize step)
export DATASET=dclm28b
export TOKENIZER=EleutherAI/pythia-12b
export SSD_MOUNT=/local/home/bingli/datasets/dclm28b/textfiles
export NFS_MOUNT=/local/home/bingli/datasets/dclm28b/tasks
export TOKENIZED_DIR=/local/home/bingli/datasets/dclm28b/tokenized
