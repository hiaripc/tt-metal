# !/bin/bash

# for i in {1..8}; do
for i in 8 16 32 64 72; do
    echo "Execution #$i .."
    export GRID_SIZE=$i
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/my_benchmark/test_grid.py::test_grid_size
    echo "--------------------------------------"
done
