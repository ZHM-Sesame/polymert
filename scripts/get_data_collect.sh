#!/bin/sh

# Google Drive file ID and desired output filename
file_id="1Ldz9qyPMyD8kYtxDx_p2tjo2x3Kj977j"
output_filename="data_collect.zip"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing gdown using pip."
    pip install gdown
fi
``
# Download the file using gdown
echo "Downloading the file..."
gdown --id "$file_id" -O "$output_filename"
echo "File downloaded successfully as $output_filename"

# Unzip the downloaded file
echo "Unzipping the file..."
unzip "$output_filename" -d data_collect
echo "File unzipped successfully to output_folder"
