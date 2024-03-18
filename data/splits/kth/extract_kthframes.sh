#!/bin/bash

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed. Please install it first."
    exit 1
fi

# Check if there are any video files in the current directory
shopt -s nullglob
video_files=(*.mp4 *.avi *.mkv)

if [ ${#video_files[@]} -eq 0 ]; then
    echo "No video files found in the current directory."
    exit 1
fi

# Loop through video files
for video_file in "${video_files[@]}"; do
    # Get the video file name without extension
    video_name=$(basename -- "${video_file%.*}")
    
    # Create a subfolder with the video name if it doesn't exist
    mkdir -p "$video_name"
    
    # Extract frames into the subfolder
    ffmpeg -i "$video_file" -vf "fps=25" "$video_name/frame%04d.png"
    
    echo "Extracted frames from $video_file and placed them in $video_name/"
done

echo "All frames extracted successfully."