import time

from loguru import logger
import csv
import pytest
import torch
import ttnn
from models.utility_functions import run_for_wormhole_b0, is_grayskull, profiler
from tests.ttnn.utils_for_testing import assert_with_pcc
from pathlib import Path
import os
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


FILE_NAME_OOB_AVG = "/home/bach/wd/nn/matmul/results/mm_oob_1_1.csv"
FILE_NAME_OOB_FR = "/home/bach/wd/nn/matmul/results/mm_oob_fr.csv"

GRID_SIZE = (8, 8)

# tilziation on device only supports bf16
# oob we are not setting anything! just the default values. MF=????
matmul_configs = [
    ("f16_m4", ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
    # ("f16_m4", ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
]

matmul_shapes_bfloat16 = [
    # (256, 256, 256),
    # (512, 512, 512),
    # (1024, 1024, 1024),
    # (2048, 2048, 2048),
    # (3072, 3072, 3072),
    # (4096, 4096, 4096),
    (8192, 8192, 8192),
]


# launch with script to reset!
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [GRID_SIZE])
@pytest.mark.parametrize("warmup_iters", [5])
@pytest.mark.parametrize("measure_iters", [100])
def  test_oob_avg(
    device,
    grid_size,
    warmup_iters,
    measure_iters,
    use_program_cache,
):

    data_info = dict.fromkeys([   
                "conf",
                "m",
                "grid_size",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "conversion_in0",
                "conversion_in1",
                "transfer_in0",
                "transfer_in1",
                "tilization_in0",
                "tilization_in1",
                "inference_avg",
                "iters",
            ])
    
    data_info['grid_size'] = grid_size
    
    # set to 0
    for k, _ in data_info.items():
        data_info[k] = 0

    data_info['iters'] = measure_iters
    def timed_exec_add(timer: str, f):
        profiler.start(timer)
        res = f()
        ttnn.synchronize_device(device)
        profiler.end(timer)
        data_info[timer] += profiler.get(timer)
        return res

    
    with open(FILE_NAME_OOB_AVG, mode="a", newline="") as file:
        writer = csv.writer(file)
        # header must be already setted
        # writer.writerow(data_info.keys())

        for conf, dtype, math_fidelity in matmul_configs:
            
            data_info['conf'] = conf
            data_info['dtype'] = dtype
            data_info['math_fidelity'] = math_fidelity

            logger.info(f"\n\nRunning conf {data_info['dtype']} ==> Type: {data_info['dtype']}, MF: {math_fidelity}\n\n")
            if data_info['dtype'] == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            else:
                raise ValueError("Metti il dtype giusto!")
            
            for m, k, n in matmul_shapes:
                data_info['m'] = m
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                logger.info(f"\nM={m} \n")

                

                # Out of box, the tensors are always in DRAM
                data_info['in0_storage_type'] = "DRAM"
                data_info['in1_storage_type'] = "DRAM"
                data_info['out_storage_type'] = "DRAM"

                #Warm up
                for i in range(warmup_iters):
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()
                    in0_t =  ttnn.from_torch(
                                in0,
                                tile=ttnn.Tile((32, 32)),
                                dtype=data_info['dtype'],
                                layout=ttnn.ROW_MAJOR_LAYOUT)
                    in0_t = ttnn.to_device(
                            in0_t, 
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    in0_t = ttnn.tilize(in0_t)

                    in1_t = ttnn.from_torch(
                            in1,
                            tile=ttnn.Tile((32, 32)),
                            dtype=data_info['dtype'],
                            layout=ttnn.ROW_MAJOR_LAYOUT)
                    in1_t = ttnn.to_device(
                            in1_t, 
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    in1_t = ttnn.tilize(in1_t)

                    output_t = in0_t @ in1_t


                tot_run = 0
                for i in range(measure_iters):
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()
                    # in0 
                    in0_t = timed_exec_add("conversion_in0",
                        lambda: ttnn.from_torch(
                            in0,
                            tile=ttnn.Tile((32, 32)),
                            dtype=data_info['dtype'],
                            layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    )

                    in0_t = timed_exec_add("transfer_in0",
                        lambda: ttnn.to_device(
                            in0_t, 
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG
                        )
                    )

                    in0_t = timed_exec_add("tilization_in0",
                        lambda: ttnn.tilize(in0_t)
                    )                

                    # in1 
                    in1_t = timed_exec_add("conversion_in1",
                        lambda: ttnn.from_torch(
                            in1,
                            tile=ttnn.Tile((32, 32)),
                            dtype=data_info['dtype'],
                            layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    )

                    in1_t = timed_exec_add("transfer_in1",
                        lambda: ttnn.to_device(
                            in1_t, 
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG
                        )
                    )

                    in1_t = timed_exec_add("tilization_in1",
                        lambda: ttnn.tilize(in1_t)
                    )  

                    # run 
                    output_t = timed_exec_add("inference_avg",
                        lambda: in0_t @ in1_t
                    )  

                ttnn.DumpDeviceProfiler(device)
                
                avg_columns = ["conversion_in0", "conversion_in1", "transfer_in0",	"transfer_in1",
                               "tilization_in0", "tilization_in1", "inference_avg"]
                for c in avg_columns:
                    data_info[c] /= measure_iters
                
                log_infos = [f"{k}: {v}" for k, v in data_info.items()]
                logger.info(f"M={m} ==> \n {log_infos}")

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(data_info.values())



# launch with script to reset!
# launch one conf at a time
# prepare pre compiled csv
# i know... i know
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [GRID_SIZE])
def  test_oob_fr(
    device,
    grid_size,
    use_program_cache,
):

    data_info = dict.fromkeys([   
                "n_run",
                "conf",
                "m",
                "grid_size",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "conversion_in0",
                "conversion_in1",
                "transfer_in0",
                "transfer_in1",
                "tilization_in0",
                "tilization_in1",
                "first_run",
                "second_run",
                "compile_time"
            ])
    n_run = os.getenv("N_FR_RUN", None)
    print(f"Test n{n_run} started..")
    assert n_run, "Set N_FR_RUN env var"
    data_info['n_run'] = n_run
    data_info['grid_size'] = grid_size


    def timed_exec(timer: str, f):
        profiler.start(timer)
        res = f()
        ttnn.synchronize_device(device)
        profiler.end(timer)
        data_info[timer] = profiler.get(timer)
        return res

    if not os.path.exists(FILE_NAME_OOB_FR):
        with open(FILE_NAME_OOB_FR, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_info.keys())
    
    with open(FILE_NAME_OOB_FR, mode="a", newline="") as file:
        writer = csv.writer(file)
        # header must be already setted
        # writer.writerow(data_info.keys())

        for conf, dtype, math_fidelity in matmul_configs:
            
            data_info['conf'] = conf
            data_info['dtype'] = dtype
            data_info['math_fidelity'] = math_fidelity

            logger.info(f"\n\nRunning conf {data_info['dtype']} ==> Type: {data_info['dtype']}, MF: {math_fidelity}\n\n")
            if data_info['dtype'] == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            else:
                raise ValueError("Metti il dtype giusto!")
            
            for m, k, n in matmul_shapes:
                data_info['m'] = m
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                logger.info(f"\nM={m} \n")

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                # Out of box, the tensors are always in DRAM
                data_info['in0_storage_type'] = "DRAM"
                data_info['in1_storage_type'] = "DRAM"
                data_info['out_storage_type'] = "DRAM"
                
                # in0 
                in0_t = timed_exec("conversion_in0",
                    lambda: ttnn.from_torch(
                        in0,
                        tile=ttnn.Tile((32, 32)),
                        dtype=data_info['dtype'],
                        layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                )

                in0_t = timed_exec("transfer_in0",
                    lambda: ttnn.to_device(
                        in0_t, 
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                )

                in0_t = timed_exec("tilization_in0",
                    lambda: ttnn.tilize(in0_t)
                )                

                # in1 
                in1_t = timed_exec("conversion_in1",
                    lambda: ttnn.from_torch(
                        in1,
                        tile=ttnn.Tile((32, 32)),
                        dtype=data_info['dtype'],
                        layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                )

                in1_t = timed_exec("transfer_in1",
                    lambda: ttnn.to_device(
                        in1_t, 
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                )

                in1_t = timed_exec("tilization_in1",
                    lambda: ttnn.tilize(in1_t)
                )  

                # run 
                output_t = timed_exec("first_run",
                    lambda: in0_t @ in1_t
                )  

                # run 
                output_t = timed_exec("second_run",
                    lambda: in0_t @ in1_t
                )   

                ttnn.DumpDeviceProfiler(device)
                
                data_info['compile_time'] = data_info['first_run'] - data_info['second_run']
                
                log_infos = [f"{k}: {v}" for k, v in data_info.items()]
                logger.info(f"M={m} ==> \n {log_infos}")

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(data_info.values())
