#!/bin/bash

mkdir framedumps

run=1

for dir in */; do
    for vid in "$dir"/processed/*; do
        ffmpeg -i "$vid" -vf fps=1 framedumps/run"$run"_frame_%04d.png
        ((run++))
    done;
done;
