#!/bin/bash

tokenize() {
    task=$1

    # Skip if task file is empty (already processed).
    [ ! -s "$task" ] && return 0

    # Read GCS link and local file path.
    link=$(sed -n '1p' $task)
    file=$(sed -n '2p' $task)

    # Tokenize the file with Megatron-LM (max 3 attempts).
    cd Megatron-LM
    for i in {1..3}; do
        echo "Tokenizing $file (Attempt $i of 3)"
        python tools/preprocess_data.py \
            --input $file \
            --output-prefix ${file%.jsonl} \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model $TOKENIZER \
            --append-eod \
            --workers ${NWORKERS:-$(nproc)} > /dev/null 2>&1 && break
        echo "Failed to tokenize $file, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to tokenize $file after 3 attempts." >&2
            return 1
        fi
    done

    # Move tokenized files to output directory.
    mkdir -p "$TOKENIZED_DIR"
    mv ${file%.jsonl}_text_document.bin ${file%.jsonl}_text_document.idx "$TOKENIZED_DIR/"

    # Mark task as completed.
    > $task
}

export -f tokenize

# Process task files with file locking to avoid conflicts.
find $NFS_MOUNT -type f -name "*.task" | while read -r line; do
    flock -n $line -c "tokenize $line" || true
done
