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

FILE_NAME_HOST_PERF = "/home/bach/wd/nn/matmul/results/mm_fp16_fg.csv"
FILE_NAME_FIRST_RUN = "/home/bach/wd/nn/matmul/results/mm_first_run_fg.csv"
FILE_NAME_OFFLOADING = "/home/bach/wd/nn/matmul/results/offloading.csv"

WARMUP_ITERS = 0
MEASURE_ITERS = 100
GRID_SIZE = (8, 8)

SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]


def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, out_sharded=False, fp32_dest_acc_en=False):
    for subblock_hw in SUBBLOCK_HW_CHOICES:
        out_subblock_h = subblock_hw[0]
        out_subblock_w = subblock_hw[1]

        if fp32_dest_acc_en:
            if (out_subblock_h * out_subblock_w) > 4:
                continue

        if out_sharded:
            if n_tiles_per_core % out_subblock_w != 0 or out_subblock_h != 1:
                continue

        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
            return (out_subblock_h, out_subblock_w)

    return (1, 1)



# This test runs different shapes for matmul_2d, with possibly the best configurations for performance.
#
# The inputs include:
#   - m, k, n: Dimensions of the input tensors.
#   - in0_sharded, out_sharded: Flags indicating whether the in0 (activation) and output tensors are sharded or not.
#   - in0_block_w_div: A parameter to divide an in0 block into multiple chunks, helping to reduce L1 cache usage.
#   - num_out_blocks_h: A parameter to divide an output block into multiple chunks on height dim, helping to reduce L1 cache usage.
#   - num_out_blocks_w: A parameter to divide an output block into multiple chunks on width dim, helping to reduce L1 cache usage.

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def   get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq

# m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w
matmul_shapes_bfloat16 = [
    (256, 256, 256, True, True, 1, 1, 1),
    (512, 512, 512, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    (3072, 3072, 3072, False, False, 2, 1, 1),
    (4096, 4096, 4096, False, False, 2, 2, 2),
    (8192, 8192, 8192, False, False, 4, 4, 4),
    # (16384, 16384, 16384, False, False, 4, 8, 8),
]

matmul_shapes_bfloat8_b = [
    (256, 256, 256, True, True, 1, 1, 1),
    (512, 512, 512, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    (3072, 3072, 3072, False, False, 2, 1, 1),
    (4096, 4096, 4096, False, False, 2, 2, 2),
    (8192, 8192, 8192, False, False, 4, 4, 4),
    # (16384, 16384, 16384, False, False, 4, 8, 8)
]

matmul_shapes_bfloat4_b = [
    (256, 256, 256, True, True, 1, 1, 1),
    (512, 512, 512, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    (3072, 3072, 3072, False, False, 2, 1, 1),
    (4096, 4096, 4096, False, False, 2, 1, 1),
    (8192, 8192, 8192, False, False, 4, 2, 2),
    # (16384, 16384, 16384, False, False, 4, 4, 4),
]

# conf, dtype, math_fidelity, use_trace 
matmul_configs = [
    ("f16_m2", ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    ("f16_m4", ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False),
    ("f8b_m2", ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False),
    ("f8b_m0", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False),
    ("f4b_m0", ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False)
]

"""
("", ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True),
("", ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True),
("", ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True),
("", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, True),
("", ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True),
"""



# @pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [GRID_SIZE])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [WARMUP_ITERS])
@pytest.mark.parametrize("num_measurement_iterations", [MEASURE_ITERS])
def  test_host_perf(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
    use_program_cache,
):

    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    data_info = dict.fromkeys([   
                "conf",
                "m",
                "grid_size",
                "in0_sharded",
                "out_sharded",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "first_run_time",
                "inference_time_avg",
                "tilization_in0",
                "tilization_in1",
                "transfer_time_in0",
                "transfer_time_in1",
                "TFLOPs (avg)",
                f"Utilization (vs {grid_size} user grid)",
                "Utilization (vs 8x11 full grid)",
            ])

    with open(FILE_NAME_HOST_PERF, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([data_info.keys()])

        for conf, dtype, math_fidelity, use_trace in matmul_configs:
            logger.info(f"\n\nRunning conf {conf} ==> Type: {dtype}, MF: {math_fidelity}, Trace: {use_trace}\n\n")
            if dtype == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            elif dtype == ttnn.bfloat8_b:
                matmul_shapes = matmul_shapes_bfloat8_b
            elif dtype == ttnn.bfloat4_b:
                matmul_shapes = matmul_shapes_bfloat4_b
            for m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w in matmul_shapes:
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m // grid_size[1] // tile_h
                per_core_N = n // grid_size[0] // tile_w
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w
                out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

                logger.info(
                    f"\nM={m} \n"
                    f"\nper_core_M: {per_core_M}, per_core_N: {per_core_N}"
                    f"\nin0_block_w_div: {in0_block_w_div}, in0_block_w: {in0_block_w}"
                    f"\nout_block_h: {out_block_h}, out_block_w: {out_block_w}"
                    f"\nout_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
                )

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                if in0_sharded:
                    data_info['in0_storage_type'] = "L1"
                else:
                    data_info['in0_storage_type'] = "DRAM"
                data_info['in1_storage_type'] = "DRAM"
                if out_sharded:
                    data_info['out_storage_type'] = "L1"
                else:
                    data_info['out_storage_type'] = "DRAM"

                if in0_sharded:
                    in0_memory_config = ttnn.create_sharded_memory_config(
                        (1, 1, m, k),
                        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                else:
                    in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                
                profiler.start(f"transfer_in0")
                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((tile_h, 32)),
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=in0_memory_config,
                )
                in0_t = ttnn.tilize(in0_t)
                profiler.end(f"offload_in0")
                
                data_info['offload_in0'] = profiler.get("offload_in0")

                profiler.start(f"transfer_in1")
                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, tile_w)),
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                profiler.end(f"transfer_in1")

                in1_t = ttnn.tilize(in1_t)

                data_info['offload_in1'] = profiler.get("offload_in1")

                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    out_block_h=out_block_h,
                    out_block_w=out_block_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                )

                compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                    math_fidelity=math_fidelity,
                    math_approx_mode=True,
                )

                if out_sharded:
                    out_mem_config = ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        buffer_type=ttnn.BufferType.L1,
                    )
                else:
                    out_mem_config = ttnn.DRAM_MEMORY_CONFIG
                if out_sharded:
                    output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
                else:
                    output_tile = ttnn.Tile([tile_h, tile_w])

                profiler.start(f"first_run_time")
                output_t = ttnn.matmul(
                    in0_t,
                    in1_t,
                    program_config=program_config,
                    memory_config=out_mem_config,
                    dtype=dtype,
                    compute_kernel_config=compute_kernel_config,
                    output_tile=output_tile,
                )
                profiler.end(f"first_run_time")

                profiler.start(f"run")
                for iter in range(0, num_measurement_iterations):
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )
                ttnn.synchronize_device(device)
                profiler.end(f"run")

                ttnn.DumpDeviceProfiler(device)
                
                data_info['first_run_time'] = profiler.get("first_run_time") 
                data_info['inference_time_avg'] = profiler.get("run") / num_measurement_iterations
                data_info['tflops'] = 2 * m * k * n / 1e12 / data_info['inference_time_avg']

                if math_fidelity == ttnn.MathFidelity.LoFi:
                    cycle_per_tile = LoFi_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi2:
                    cycle_per_tile = HiFi2_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi3:
                    cycle_per_tile = HiFi3_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi4:
                    cycle_per_tile = HiFi4_cycle
                num_cores_user_grid = grid_size[0] * grid_size[1]
                compute_grid_size = device.compute_with_storage_grid_size()
                num_cores_full_grid = compute_grid_size.x * compute_grid_size.y
                ideal_cycle_full_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_full_grid
                ideal_cycle_user_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_user_grid
                inference_cycle = data_info['inference_time_avg'] * get_device_freq() * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                data_info['utilization_full_grid_percentage'] = f"{utilization_full_grid * 100:.2f}%"
                data_info['utilization_user_grid_percentage'] = f"{utilization_user_grid * 100:.2f}%"

                logger.info(f"M={m} ==> \n {[f"{k}: {v}" for k,v in data_info.items()]}")

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(data_info.values())



# @pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [GRID_SIZE])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [WARMUP_ITERS])
@pytest.mark.parametrize("num_measurement_iterations", [MEASURE_ITERS])
def test_first_run(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
    use_program_cache
):

    logger.info(f"\n\nStarting test...")
    with open(FILE_NAME_FIRST_RUN, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([   
                "conf",
                "m",
                "use_trace",
                "grid_size",
                "in0_sharded",
                "out_sharded",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "kernel_config_time",
                "first_run_time",
                "second_run_time",
                "compile_time",
                "transfer_time_in0",
                "transfer_time_in1",
            ]
        )
        
        ttnn.disable_and_clear_program_cache(device)
        # ttnn.close_device(device)

        for conf, dtype, math_fidelity, use_trace in matmul_configs:
            logger.info(f"\n\nRunning conf {conf} ==> Type: {dtype}, MF: {math_fidelity}, Trace: {use_trace}\n\n")
            if dtype == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            elif dtype == ttnn.bfloat8_b:
                matmul_shapes = matmul_shapes_bfloat8_b
            elif dtype == ttnn.bfloat4_b:
                matmul_shapes = matmul_shapes_bfloat4_b
            for m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w in matmul_shapes:
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m // grid_size[1] // tile_h
                per_core_N = n // grid_size[0] // tile_w
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w
                out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

                logger.info(f"\nM:{m} out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")

                kernel_config_time_acc = 0
                first_run_time_acc = 0
                second_run_time_acc = 0
                offload_in0_acc = 0
                offload_in1_acc = 0
                
                for it in range(num_measurement_iterations):
                    # device = ttnn.open_device(device_id=0)
                    ttnn.disable_and_clear_program_cache(device)
                    # ttnn.enable_program_cache(device)
  
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()

                    if in0_sharded:
                        in0_storage_type = "L1"
                    else:
                        in0_storage_type = "DRAM"
                    in1_storage_type = "DRAM"
                    if out_sharded:
                        out_storage_type = "L1"
                    else:
                        out_storage_type = "DRAM"

                    if in0_sharded:
                        in0_memory_config = ttnn.create_sharded_memory_config(
                            (1, 1, m, k),
                            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                            strategy=ttnn.ShardStrategy.BLOCK,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        )
                    else:
                        in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                    
                    profiler.start(f"offload_in0")
                    in0_t = ttnn.from_torch(
                        in0,
                        tile=ttnn.Tile((tile_h, 32)),
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=in0_memory_config,
                    )
                    profiler.end(f"offload_in0")
                    

                    profiler.start(f"offload_in1")
                    in1_t = ttnn.from_torch(
                        in1,
                        tile=ttnn.Tile((32, tile_w)),
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    profiler.end(f"offload_in1")

                    profiler.start(f"kernel_config_time")
                    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=grid_size,
                        in0_block_w=in0_block_w,
                        out_subblock_h=out_subblock_h,
                        out_subblock_w=out_subblock_w,
                        out_block_h=out_block_h,
                        out_block_w=out_block_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=False,
                        fused_activation=None,
                    )

                    compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                        math_fidelity=math_fidelity,
                        math_approx_mode=True,
                    )

                    if out_sharded:
                        out_mem_config = ttnn.MemoryConfig(
                            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            buffer_type=ttnn.BufferType.L1,
                        )
                    else:
                        out_mem_config = ttnn.DRAM_MEMORY_CONFIG
                    if out_sharded:
                        output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
                    else:
                        output_tile = ttnn.Tile([tile_h, tile_w])
                    profiler.end(f"kernel_config_time")

                    profiler.start(f"first_run_time")
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )
                    ttnn.synchronize_device(device)
                    profiler.end(f"first_run_time")


                    profiler.start(f"second_run_time")
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )
                    ttnn.synchronize_device(device)
                    profiler.end(f"second_run_time")

                    ttnn.DumpDeviceProfiler(device)
                    
                    output_tensor = ttnn.to_torch(output_t)
                    ttnn.deallocate(output_t)
                    ttnn.deallocate(in0_t)
                    ttnn.deallocate(in1_t)

                    kernel_config_time = profiler.get("kernel_config_time")
                    first_run_time = profiler.get("first_run_time")
                    second_run_time = profiler.get("second_run_time")
                    offload_in0 = profiler.get("offload_in0")
                    offload_in1 = profiler.get("offload_in1")

                    logger.info(f"\nIT {it} => FR: {first_run_time}, SR: {second_run_time}, T1: {offload_in0}, T2: {offload_in1}")

                    kernel_config_time_acc += kernel_config_time
                    first_run_time_acc += first_run_time
                    second_run_time_acc += second_run_time
                    offload_in0_acc += offload_in0
                    offload_in1_acc += offload_in1 
                    
                    # ttnn.
                    # ttnn.enable_program_cache(device)
                    # ttnn.close_device(device)

                # Retrive data and write 
                kernel_config_time_acc /= num_measurement_iterations
                first_run_time_acc /= num_measurement_iterations
                second_run_time_acc /= num_measurement_iterations
                offload_in0_acc /= num_measurement_iterations
                offload_in1_acc /= num_measurement_iterations
                compile_time = first_run_time_acc - second_run_time_acc
                
                
                writer.writerow([   
                        conf,
                        m,
                        f"{True}" if use_trace else f"{False}",
                        grid_size,
                        in0_sharded,
                        out_sharded,
                        in0_storage_type,
                        in1_storage_type,
                        out_storage_type,
                        dtype,
                        math_fidelity,
                        f"{kernel_config_time_acc * 1e6:.2f}",
                        f"{first_run_time_acc * 1e6:.2f}",
                        f"{second_run_time_acc * 1e6:.2f}",
                        f"{compile_time * 1e6:.2f}",
                        f"{offload_in0_acc * 1e6:.2f}",
                        f"{offload_in1_acc * 1e6:.2f}",
                    ]
                )

@pytest.mark.parametrize("grid_size", [(8, 8)])
@pytest.mark.parametrize("num_measurement_iterations", [MEASURE_ITERS])
def test_offloading(
    device,
    grid_size,
    num_measurement_iterations
):
    logger.info(f"\n\nStarting test...")
    with open(FILE_NAME_OFFLOADING, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([   
                "conf",
                "m",
                "grid_size",
                "in0_sharded",
                "out_sharded",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "torch_creation",
                "ttnn_creation",
                "to_layout",
                "to_device",
        ])
        
        for conf, dtype, math_fidelity, use_trace in matmul_configs:
            logger.info(f"\n\nRunning conf {conf} ==> Type: {dtype}, MF: {math_fidelity}, Trace: {use_trace}\n\n")
            if dtype == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            elif dtype == ttnn.bfloat8_b:
                matmul_shapes = matmul_shapes_bfloat8_b
            elif dtype == ttnn.bfloat4_b:
                matmul_shapes = matmul_shapes_bfloat4_b
            for m, k, n, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w in matmul_shapes:
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m // grid_size[1] // 32
                per_core_N = n // grid_size[0] // 32
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w
                out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

                logger.info(f"\nM:{m} out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                if in0_sharded:
                    in0_storage_type = "L1"
                else:
                    in0_storage_type = "DRAM"
                in1_storage_type = "DRAM"
                if out_sharded:
                    out_storage_type = "L1"
                else:
                    out_storage_type = "DRAM"

                if in0_sharded:
                    in0_memory_config = ttnn.create_sharded_memory_config(
                        (1, 1, m, k),
                        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                else:
                    in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                
                torch_creation_acc = 0
                ttnn_creation_acc = 0
                to_layout_acc = 0
                to_device_acc = 0
                for it in range(num_measurement_iterations):
                    profiler.start("torch_tensor_creation")
                    in_layout_torch = torch.ones(in0_shape).bfloat16()
                    profiler.end("torch_tensor_creation")
                    torch_creation = profiler.get("torch_tensor_creation")
                    torch_creation_acc += torch_creation

                    profiler.start(f"ttnn_tensor_creation")
                    in_layout_t = ttnn.ones(
                        in0_shape, 
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    profiler.end(f"ttnn_tensor_creation")
                    ttnn_creation = profiler.get(f"ttnn_tensor_creation")
                    ttnn_creation_acc += ttnn_creation

                    profiler.start(f"to_layout")
                    in_layout_t = ttnn.to_layout(
                        in_layout_t,
                        layout=ttnn.TILE_LAYOUT
                    )
                    profiler.end(f"to_layout")
                    to_layout = profiler.get("to_layout")
                    to_layout_acc += to_layout

                    profiler.start(f"to_device")
                    in_layout_t = ttnn.to_device(
                        in_layout_t, 
                        device=device, 
                        memory_config=in0_memory_config)
                    profiler.end(f"to_device")
                    to_device = profiler.get("to_device")
                    to_device_acc += to_device

                    logger.info(f"\nIT{it} => torch_creation: {torch_creation}, ttnn_creation: {ttnn_creation}, to_layout: {to_layout}, to_device: {to_device}")

                torch_creation_acc = torch_creation_acc/num_measurement_iterations
                ttnn_creation_acc = ttnn_creation_acc/num_measurement_iterations
                to_layout_acc = to_layout_acc/num_measurement_iterations
                to_device_acc = to_device_acc/num_measurement_iterations
                
                writer.writerow([   
                        conf,
                        m,
                        grid_size,
                        in0_sharded,
                        out_sharded,
                        in0_storage_type,
                        in1_storage_type,
                        out_storage_type,
                        dtype,
                        math_fidelity,
                        f"{torch_creation_acc * 1e6:.2f}",
                        f"{ttnn_creation_acc * 1e6:.2f}",
                        f"{to_layout * 1e6:.2f}",
                        f"{to_device * 1e6:.2f}",
                    ]
                )
    pass


@pytest.mark.parametrize("grid_size", [(8, 8)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [WARMUP_ITERS])
@pytest.mark.parametrize("num_measurement_iterations", [MEASURE_ITERS])
def test_dummy(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
    use_program_cache
):
    pass