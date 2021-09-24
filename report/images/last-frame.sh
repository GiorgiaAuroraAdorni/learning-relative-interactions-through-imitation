#!/bin/bash

input=$1
output=$2

rm "$output"

n_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 $input)

ffmpeg -i "$input" -vf "select='eq(n, $n_frames - 1)'" -vframes 1 "$output"
