#!/bin/bash
# Run all example scripts and save outputs to kagome_ex_results/

set -e

cd /home/oakk/Projects/condmatTensor260124/condmatTensor/examples
PYTHONPATH=/home/oakk/Projects/condmatTensor260124/condmatTensor/src:$PYTHONPATH
PYTHON=/home/oakk/Projects/condmatTensor260124/condmatTensor/env_condmatTensor/bin/python

echo "========================================"
echo "Running all example scripts..."
echo "========================================"

# List all Python files (excluding cache and special files)
for script in *.py; do
    if [ "$script" != "__init__.py" ] && [ -f "$script" ]; then
        echo ""
        echo "===== Running $script ====="
        $PYTHON "$script" || echo "  (exited with code $?)"
    fi
done

echo ""
echo "========================================"
echo "All examples completed!"
echo "========================================"
echo ""
echo "Generated files:"
ls -1 *.png 2>/dev/null || echo "No PNG files found"

