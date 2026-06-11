#!/bin/bash
# View evaluation results

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "============================================"
echo "EVALUATION RESULTS SUMMARY"
echo "============================================"
echo ""

echo "Test Set Results:"
echo "----------------"
cat results/test_results.txt 2>/dev/null || echo "No test results found"
echo ""

echo "Validation Set Results:"
echo "----------------------"
cat results/val_results.txt 2>/dev/null || echo "No validation results found"
echo ""

echo "Training Set Results:"
echo "--------------------"
cat results/train_results.txt 2>/dev/null || echo "No training results found"
echo ""

echo "============================================"
echo "Confusion Matrix Images:"
echo "============================================"
ls -lh results/*_confusion_matrix.png 2>/dev/null || echo "No confusion matrices found"
echo ""

echo "To view confusion matrices, open:"
echo "  - results/test_confusion_matrix.png"
echo "  - results/val_confusion_matrix.png"
echo "  - results/train_confusion_matrix.png"



