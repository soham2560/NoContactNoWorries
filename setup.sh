#!/bin/bash

# Install gdown (if not already installed)
if ! command -V gdown &> /dev/null
then
  echo "gdown not found. Installing..."
  pip install gdown
  if [ $? -ne 0 ]; then
    echo "Error installing gdown.  Exiting."
    exit 1
  fi
else
  echo "gdown already installed."
fi

# Set the Google Drive ID
GOOGLE_DRIVE_ID="1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR"  # Replace with your actual ID

# Download the Asymformer .pth file using gdown
echo "Downloading .pth file..."
gdown "$GOOGLE_DRIVE_ID" --output checkpoints/ 
if [ $? -ne 0 ]; then
  echo "Error downloading file.  Exiting."
  exit 1
fi

echo "Done."
