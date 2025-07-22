#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Configuration ---
ENV_NAME="dro"

# --- Main Script ---
echo "Creating micromamba environment: $ENV_NAME"
micromamba create -n $ENV_NAME -y

echo "Installing packages from conda-forge and bioconda..."
# Run the installation commands within the created environment
micromamba run -n $ENV_NAME mamba install -c conda-forge -c bioconda bart -y

echo "Installing packages using pip..."
micromamba run -n $ENV_NAME pip install \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    sigpy \
    imageio \
    torch \
    torchkbnufft \
    nibabel \
    einops

echo "Environment '$ENV_NAME' created and packages installed successfully."
echo "To activate the environment, run: micromamba activate $ENV_NAME"