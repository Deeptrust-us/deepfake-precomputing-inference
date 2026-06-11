#!/bin/bash
# Quad-Stream Deepfake Detection Pipeline Runner

set -e

echo "============================================"
echo "QUAD-STREAM DEEPFAKE DETECTION PIPELINE"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "config/config.yaml" ]; then
    echo "Error: config/config.yaml not found. Please run from project root."
    exit 1
fi

# Step 1: Preprocessing
echo -e "${GREEN}Step 1: Preprocessing dataset...${NC}"
python scripts/preprocess.py --max_videos 10

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 2: Training model...${NC}"
echo "This may take a while. Training logs will be saved to training_output.log"
echo ""

# Step 2: Training
python scripts/train.py --config config/config.yaml 2>&1 | tee training_output.log

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 3: Evaluating model...${NC}"

# Step 3: Evaluation
if [ -f "checkpoints/best_model.pth" ]; then
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
else
    echo -e "${YELLOW}Warning: No best_model.pth found. Using latest checkpoint.${NC}"
    python scripts/evaluate.py --checkpoint checkpoints/latest.pth --split test
fi

echo ""
echo -e "${GREEN}============================================"
echo "Pipeline completed successfully!"
echo "============================================${NC}"
echo ""
echo "Results saved to: results/"
echo "Checkpoints saved to: checkpoints/"
echo "Training logs: training_output.log"
echo ""
echo "To view training progress: tensorboard --logdir logs"


