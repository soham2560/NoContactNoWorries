# Docker Image for Model Training with Huggingface 🤗 Accelerate 

This directory contains scripts to build a Docker image for training machine learning models using Huggingface 🤗 Accelerate. The image is minimal and does not include support for the simulator.

## Contents

- `Dockerfile`: Instructions to build the Docker image.
- `build.sh`: Script to build the Docker image.
- `README.md`: This readme file.

## Usage

1. Build the Docker image:
    ```sh
    ./build.sh
    ```

2. Run the Docker container:
    ```sh
    docker run -it <image_name>
    ```

## Notes

- This Docker image is designed for training models and does not support the simulator.
- Ensure you have Docker installed and running on your system.

For more information on Huggingface Accelerate, visit the [official documentation](https://huggingface.co/docs/accelerate/index).
