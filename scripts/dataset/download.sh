#!/bin/bash
# Download the desired dataset.
# Usage: scripts/dataset/download.sh

# Author: Hao Kang
# Date: March 9, 2025

source devconfig.sh

export SSD_MOUNT=/local/home/bingli/datasets/dclm28b/textfiles

export NFS_MOUNT=/local/home/bingli/datasets/dclm28b/tasks

export TOKENIZED_DIR=/local/home/bingli/datasets/dclm28b/tokenized


download_dclm28b() {
    # Each task file contains the S3 link and the local file path.
    prefix=s3://commoncrawl/contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/
    base_url=https://data.commoncrawl.org/contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/
    curl -s "https://commoncrawl.s3.amazonaws.com/?list-type=2&prefix=contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/" \
        | grep -o '<Key>[^<]*</Key>' | sed 's/<Key>//;s/<\/Key>//' \
        | while read -r key; do
        name=$(basename $key)
        link=$base_url$name
        name=gs0310-ls110_$name
        file=$SSD_MOUNT/$name
        task=$NFS_MOUNT/$name.task
        echo $link >> $task
        echo $file >> $task
    done
    # Dispatch the tasks to the nodes.
    bash scripts/dataset/modules/download_dclm_step1.sh
}

case $DATASET in
    dclm28b)
        download_dclm28b
        ;;
    *)
        echo "Unknown dataset: $1"
        exit 1
esac
