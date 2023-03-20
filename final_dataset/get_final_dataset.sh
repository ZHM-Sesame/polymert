#!/bin/sh

# Google Drive file ID and desired output filename
file_id="17aDPxsuEtF1i4NDhrC9DSWgb12YGoBnl"
output_filename="final_dataset.zip"

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
unzip "$output_filename"
echo "File unzipped successfully to output_folder"