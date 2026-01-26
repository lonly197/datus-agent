#!/bin/bash

# Script to download HuggingFace models for domestic environment (China)
# Supports HuggingFace mirror and ModelScope fallback

set -e

# Configuration
MODELS_DIR="${HOME}/.datus/models"
HF_MIRROR="https://hf-mirror.com"
MODELSCOPE_URL="https://modelscope.cn/models"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check and install dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    pip install huggingface_hub hf_transfer modelscope -q 2>/dev/null || true
}

# Download from HuggingFace (with mirror support)
download_from_hf() {
    local model_path=$1
    local model_name=$(basename "$model_path")
    local target_dir="${MODELS_DIR}/${model_name}"

    log_info "Downloading ${model_path} from HuggingFace..."

    # Try HuggingFace mirror first
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_ENDPOINT="${HF_MIRROR}"

    mkdir -p "${target_dir}"

    if hf download --resume-download "${model_path}" \
        --local-dir "${target_dir}" \
        --local-dir-use-symlinks False 2>/dev/null; then
        log_info "Downloaded ${model_name} successfully!"
        return 0
    fi

    # Fallback to original HuggingFace
    unset HF_ENDPOINT
    log_warn "Mirror failed, trying original HuggingFace..."
    if hf download --resume-download "${model_path}" \
        --local-dir "${target_dir}" \
        --local-dir-use-symlinks False 2>/dev/null; then
        log_info "Downloaded ${model_name} successfully!"
        return 0
    fi

    log_error "Failed to download ${model_name}"
    return 1
}

# Download from ModelScope
download_from_modelscope() {
    local model_path=$1
    local model_name=$(basename "$model_path")
    local target_dir="${MODELS_DIR}/${model_name}"

    log_info "Downloading ${model_name} from ModelScope..."

    mkdir -p "${target_dir}"

    # ModelScope uses different namespace, need to map
    # BAAI/bge-reranker-large -> XorML/bge-reranker-large
    case "$model_path" in
        "BAAI/bge-reranker-large")
            pip install ms-swift -q 2>/dev/null
            python3 -c "
from swift import snapshot_download
snapshot_download(
    repo_id='XorML/bge-reranker-large',
    local_dir='${target_dir}',
    local_dir_use_symlinks=False
)
"
            ;;
        "qdrant/all-MiniLM-L6-v2-onnx")
            pip install ms-swift -q 2>/dev/null
            python3 -c "
from swift import snapshot_download
snapshot_download(
    repo_id='qdrant/qdrant--all-MiniLM-L6-v2-onnx',
    local_dir='${target_dir}',
    local_dir_use_symlinks=False
)
"
            ;;
        *)
            log_error "No ModelScope equivalent for ${model_path}"
            return 1
            ;;
    esac

    log_info "Downloaded ${model_name} from ModelScope!"
}

# Main download function with fallback
download_model() {
    local model_path=$1
    local prefer_mirror=${2:-"hf"}  # "hf" or "ms"

    log_info "Downloading model: ${model_path}"

    if [ "$prefer_mirror" = "hf" ]; then
        if ! download_from_hf "$model_path"; then
            log_warn "HuggingFace download failed, trying ModelScope..."
            download_from_modelscope "$model_path" || log_error "All download methods failed"
        fi
    else
        if ! download_from_modelscope "$model_path"; then
            log_warn "ModelScope download failed, trying HuggingFace..."
            download_from_hf "$model_path" || log_error "All download methods failed"
        fi
    fi
}

# Main
main() {
    echo "============================================"
    echo "  Datus Agent - Model Download Script"
    echo "  (Domestic Environment Support)"
    echo "============================================"
    echo ""

    check_dependencies

    # Create models directory
    mkdir -p "${MODELS_DIR}"

    # Core embedding model (required)
    echo ""
    log_info "Step 1/2: Downloading core embedding model..."
    download_model "qdrant/all-MiniLM-L6-v2-onnx"

    # Reranker model (optional, for hybrid search enhancement)
    echo ""
    log_info "Step 2/2: Downloading reranker model (optional)..."
    log_info "  Note: This model is ~1.5GB and only needed when hybrid_rerank_enabled=true"
    read -p "  Download reranker model? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_model "BAAI/bge-reranker-large"
    else
        log_info "Skipping reranker model download"
    fi

    echo ""
    echo "============================================"
    log_info "Download complete!"
    log_info "Models saved to: ${MODELS_DIR}"
    echo "============================================"
}

# Run main
main "$@"
