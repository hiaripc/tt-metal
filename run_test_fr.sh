# !/bin/bash

for i in {1..100}; do
    echo "Execution #$i .."
    export N_FR_RUN=$i
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/my_benchmark/test_oob.py::test_oob_fr
    echo "--------------------------------------"
done
