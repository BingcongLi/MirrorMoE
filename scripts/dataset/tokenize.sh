#!/bin/bash

# Setup the runtime environment.
source devconfig.sh

# Grab all the files to be tokenized and store into the parameter queue.
find "$SSD_MOUNT" -type f -name "*.jsonl" | while read -r line; do
    name=$(basename $line)
    task=$NFS_MOUNT/$name.task
    echo $line >> $task
    echo $line >> $task
done

# Dispatch the tokenization.
bash scripts/dataset/modules/tokenize_step1.sh
