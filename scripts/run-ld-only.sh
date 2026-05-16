#!/bin/bash

INPUT_DIR="$1"
OUTPUT_FILE="adas_syslog_entries.log"

touch $OUTPUT_FILE

START="$(date '+%Y-%m-%d %H:%M:%S')"

for vid in "$INPUT_DIR"/batch-*/720p/*; do
    ./build/adas "$vid" -disable-sign -disable-ped
done;

END="$(date '+%Y-%m-%d %H:%M:%S')"

journalctl --since "$START" --until "$END" | grep "AV/ADAS Metrics" >> "$OUTPUT_FILE"

echo "Saved ADAS syslog entries to $OUTPUT_FILE"
