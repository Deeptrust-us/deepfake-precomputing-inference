#!/bin/bash
# Start training script

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "Starting training..."
echo ""


# Start training in screen session
screen -S deep bash -c "cd '$PROJECT_ROOT' && python scripts/train.py --config config/config.yaml 2>&1 | tee training_output.log"

echo ""
echo "Training started in screen session 'deep'"
echo ""
echo "To attach: screen -r deep"
echo "To detach: Press Ctrl+A then D"
echo "To view logs: tail -f training_output.log"




