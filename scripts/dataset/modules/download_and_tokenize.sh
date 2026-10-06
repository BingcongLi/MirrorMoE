#!/bin/bash
# Download, tokenize, and delete each raw file immediately to save disk space.
# Invoked by scripts/dataset/prepare.sh

source devconfig.sh


process() {
    task=$1

    # Skip if task file is empty (already processed).
    [ ! -s "$task" ] && return 0

    # Read link and local file path.
    link=$(sed -n '1p' $task)
    file=$(sed -n '2p' $task)

    # ── Download ──────────────────────────────────────────────────────────────
    for i in {1..3}; do
        echo "[$(date +%H:%M:%S)] Downloading $link (Attempt $i of 3)"
        wget -q "$link" -O "$file" && break
        echo "Failed to download, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to download $link after 3 attempts." >&2
            return 1
        fi
    done

    # ── Extract .zstd ─────────────────────────────────────────────────────────
    for i in {1..3}; do
        echo "[$(date +%H:%M:%S)] Extracting $file (Attempt $i of 3)"
        unzstd "$file" -o "${file%.zst}" > /dev/null 2>&1 && rm -f "$file" && break
        echo "Failed to extract, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to extract $file after 3 attempts." >&2
            return 1
        fi
    done
    jsonl_file="${file%.zst}"

    # ── Tokenize ──────────────────────────────────────────────────────────────
    cd Megatron-LM
    for i in {1..3}; do
        echo "[$(date +%H:%M:%S)] Tokenizing $jsonl_file (Attempt $i of 3)"
        python tools/preprocess_data.py \
            --input "$jsonl_file" \
            --output-prefix "${jsonl_file%.jsonl}" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER" \
            --append-eod \
            --workers "${NWORKERS:-$(nproc)}" > /dev/null 2>&1 && break
        echo "Failed to tokenize, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to tokenize $jsonl_file after 3 attempts." >&2
            rm -f "$jsonl_file"
            return 1
        fi
    done
    cd - > /dev/null

    # ── Move tokenized files and delete raw ───────────────────────────────────
    mkdir -p "$TOKENIZED_DIR"
    mv "${jsonl_file%.jsonl}_text_document.bin" \
       "${jsonl_file%.jsonl}_text_document.idx" \
       "$TOKENIZED_DIR/"
    rm -f "$jsonl_file"
    echo "[$(date +%H:%M:%S)] Done: $(basename $jsonl_file)"

    # Mark task as completed.
    > "$task"
}

export -f process

# Process task files with file locking to avoid conflicts.
find "$NFS_MOUNT" -type f -name "*.task" | while read -r line; do
    flock -n "$line" -c "process $line" || true
done
