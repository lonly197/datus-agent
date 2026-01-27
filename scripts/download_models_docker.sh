#!/bin/bash

# Script to download required HuggingFace models for Docker build
# This script downloads:
# 1. qdrant/all-MiniLM-L6-v2-onnx - Core embedding model (required)
# 2. BAAI/bge-reranker-large - Reranker model (optional, for hybrid search)

set -e

echo "Installing required tools..."
pip install huggingface_hub hf_transfer

echo "Setting up high-speed transfer..."
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Creating model directories..."
mkdir -p docker/huggingface/fastembed
mkdir -p docker/huggingface/reranker

echo "Downloading qdrant/all-MiniLM-L6-v2-onnx model (embedding)..."
hf download --resume-download qdrant/all-MiniLM-L6-v2-onnx \
  --local-dir docker/huggingface/fastembed/qdrant--all-MiniLM-L6-v2-onnx \
  --local-dir-use-symlinks False

echo "Downloading BAAI/bge-reranker-large model (reranker - optional)..."
# Only download if the model repo exists and is accessible
if hf repo info BAAI/bge-reranker-large &>/dev/null; then
  hf download --resume-download BAAI/bge-reranker-large \
    --local-dir docker/huggingface/reranker/BAAI--bge-reranker-large \
    --local-dir-use-symlinks False
  echo "Reranker model downloaded successfully!"
else
  echo "Warning: Reranker model not available, skipping..."
  echo "  Note: This model is only needed when hybrid_rerank_enabled=true"
fi

echo "Model download completed!"
echo "You can now build the Docker image with: docker build -t datus-agent:latest ."