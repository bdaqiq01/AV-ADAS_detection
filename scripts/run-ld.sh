!/bin/bash
INPUT_DIR="$1"
OUTPUT_DIR="$2"

mkdir $OUTPUT_DIR

count=0
for input_file in "$INPUT_DIR"/*; do
    output_file="$OUTPUT_DIR/output_$count.mp4"

    echo "Processing $input_file"
    echo "Debug 1: $INPUT_DIR/$input_file"
    echo "Debug 2: $OUTPUT_DIR/$output_file"

    ./build/adas $input_file -o -disable-ped -disable-sign
    mv output/final_output.mp4 $output_file

    ((count++))
done

