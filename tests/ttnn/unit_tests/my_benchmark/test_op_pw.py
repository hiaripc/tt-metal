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
from .tt_smi_analyzer import TelemetryAnalyzer
import numpy as np

ta = TelemetryAnalyzer()
profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


WARMUP_ITERS = 5
MEASURE_ITERS = 300
GRID_SIZE = (8, 8)

FILE_NAME = "/home/bach/wd/nn/matmul/results/op_pw_desperate.csv"
FILE_NAME_OOB = "/home/bach/wd/nn/matmul/results/oob_pw.csv"

# tilziation on device only supports bf16
matmul_configs = [
    #  ("f32_m4", ttnn.float32, ttnn.MathFidelity.HiFi4),
    ("f16_m2", ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
    ("f16_m4", ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
    # ("f8b_m2", ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
    # ("f8b_m0", ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    # ("f4b_m0", ttnn.bfloat4_b, ttnn.MathFidelity.LoFi)
]

matmul_shapes_bfloat16 = [
    # (256, 256, 256, True, True, 1, 1, 1),
    (512, 512, 512, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    # (3072, 3072, 3072, False, False, 2, 1, 1),
    (4096, 4096, 4096, False, False, 2, 2, 2),
    (8192, 8192, 8192, False, False, 4, 4, 4),
]


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


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("warmup_iters", [WARMUP_ITERS])
@pytest.mark.parametrize("measure_iters", [MEASURE_ITERS])
@pytest.mark.parametrize("grid_size", [GRID_SIZE])
def test_op(
    device,
    warmup_iters,
    measure_iters,
    grid_size,
    use_program_cache,
):
    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    data_info = dict()

    conf_infos = [
        "conf",
        "m",
        "grid_size",
        "in0_storage_type",
        "in1_storage_type",
        "out_storage_type",
        "dtype",
        "math_fidelity",
        "utilization_vs_user_grid_perc",
        "utilization_vs_full_grid_perc",
    ]

    timing_infos = ["run", "tflops"]

    telemetry_infos = ["voltage", "current", "power", "aiclk", "temp"]

    for k in conf_infos + timing_infos + telemetry_infos:
        data_info[k] = None

    data_info["grid_size"] = grid_size
    grid_y, grid_x = grid_size

    data_info["iters"] = measure_iters

    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_info.keys())

    with open(FILE_NAME, mode="a", newline="") as file:
        writer = csv.writer(file)

        for conf, dtype, math_fidelity in matmul_configs:
            data_info["conf"] = conf
            data_info["dtype"] = dtype
            data_info["math_fidelity"] = math_fidelity

            logger.info(
                f"\n\nRunning conf {data_info['dtype']} ==> Type: {data_info['dtype']}, MF: {math_fidelity}\n\n"
            )

            for (
                m,
                k,
                n,
                in0_sharded,
                out_sharded,
                in0_block_w_div,
                num_out_blocks_h,
                num_out_blocks_w,
            ) in matmul_shapes_bfloat16:
                data_info["m"] = m
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m // grid_size[1] // 32
                per_core_N = n // grid_size[0] // 32
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w
                out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

                if in0_sharded:
                    data_info["in0_storage_type"] = "L1"
                else:
                    data_info["in0_storage_type"] = "DRAM"
                data_info["in1_storage_type"] = "DRAM"
                if out_sharded:
                    data_info["out_storage_type"] = "L1"
                else:
                    data_info["out_storage_type"] = "DRAM"

                if in0_sharded:
                    in0_memory_config = ttnn.create_sharded_memory_config(
                        (1, 1, m, k),
                        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                else:
                    in0_memory_config = ttnn.DRAM_MEMORY_CONFIG

                logger.info(f"\nM={m} \n")

                # set to 0
                for i in timing_infos:
                    data_info[i] = 0

                for i in telemetry_infos:
                    data_info[i] = 0

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
                    output_tile = ttnn.Tile([32, 32]) if 32 <= 16 else ttnn.Tile([32, 32])
                else:
                    output_tile = ttnn.Tile([32, 32])

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()
                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((32, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=in0_memory_config,
                )

                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # Warm up
                for i in range(warmup_iters):
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )

                profiler.start(f"run")
                ta.start_monitoring_online(freq=1)
                for i in range(measure_iters):
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
                telemetry_data = ta.stop_monitoring_online()

                ttnn.DumpDeviceProfiler(device)

                for i, v in telemetry_data.items():
                    data_info[i] += np.mean(v)

                inference_time_avg = profiler.get("run") / measure_iters
                data_info["tflops"] = 2 * m * k * n / 1e12 / inference_time_avg
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
                ideal_cycle_full_grid = m * k * n / 32 / 32 / 32 * cycle_per_tile / num_cores_full_grid
                ideal_cycle_user_grid = m * k * n / 32 / 32 / 32 * cycle_per_tile / num_cores_user_grid
                inference_cycle = inference_time_avg * get_device_freq() * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                utilization_full_grid_percentage = f"{utilization_full_grid * 100:.2f}%"
                utilization_user_grid_percentage = f"{utilization_user_grid * 100:.2f}%"

                data_info["run"] = inference_time_avg * 1e6
                data_info["utilization_vs_full_grid_perc"] = utilization_full_grid_percentage
                data_info["utilization_vs_user_grid_perc"] = utilization_user_grid_percentage

                log_infos = [f"{k}: {v}" for k, v in data_info.items()]
                logger.info(f"\nM={m} ==> \n {log_infos}")

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(data_info.values())

                device.disable_and_clear_program_cache()
                device.enable_program_cache()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("warmup_iters", [WARMUP_ITERS])
@pytest.mark.parametrize("measure_iters", [MEASURE_ITERS])
@pytest.mark.parametrize("grid_size", [GRID_SIZE])
def test_oob(
    device,
    warmup_iters,
    measure_iters,
    grid_size,
    use_program_cache,
):
    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    data_info = dict()

    conf_infos = [
        "conf",
        "m",
        "grid_size",
        "in0_storage_type",
        "in1_storage_type",
        "out_storage_type",
        "dtype",
        "math_fidelity",
        "utilization_vs_user_grid_perc",
        "utilization_vs_full_grid_perc",
    ]

    timing_infos = ["run", "tflops"]

    telemetry_infos = ["voltage", "current", "power", "aiclk", "temp"]

    for k in conf_infos + timing_infos + telemetry_infos:
        data_info[k] = None

    data_info["grid_size"] = grid_size
    grid_y, grid_x = grid_size

    data_info["iters"] = measure_iters

    if not os.path.exists(FILE_NAME_OOB):
        with open(FILE_NAME_OOB, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_info.keys())

    with open(FILE_NAME_OOB, mode="a", newline="") as file:
        writer = csv.writer(file)

        for conf, dtype, math_fidelity in matmul_configs:
            data_info["conf"] = conf
            data_info["dtype"] = dtype
            data_info["math_fidelity"] = math_fidelity

            logger.info(
                f"\n\nRunning conf {data_info['dtype']} ==> Type: {data_info['dtype']}, MF: {math_fidelity}\n\n"
            )

            for (
                m,
                k,
                n,
                in0_sharded,
                out_sharded,
                in0_block_w_div,
                num_out_blocks_h,
                num_out_blocks_w,
            ) in matmul_shapes_bfloat16:
                data_info["m"] = m
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                logger.info(f"\nM={m} \n")

                # set to 0
                for i in timing_infos:
                    data_info[i] = 0

                for i in telemetry_infos:
                    data_info[i] = 0

                compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                    math_fidelity=math_fidelity,
                    math_approx_mode=True,
                )

                data_info["in0_storage_type"] = "DRAM"
                data_info["in1_storage_type"] = "DRAM"
                data_info["out_storage_type"] = "DRAM"

                in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                in1_memory_config = ttnn.DRAM_MEMORY_CONFIG
                out_mem_config = ttnn.DRAM_MEMORY_CONFIG
                output_tile = ttnn.Tile([32, 32])

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((32, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=in0_memory_config,
                )

                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=in1_memory_config,
                )

                # Warm up
                for i in range(warmup_iters):
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )

                profiler.start(f"run")
                ta.start_monitoring_online(freq=1)
                for i in range(measure_iters):
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )
                ttnn.synchronize_device(device)
                profiler.end(f"run")
                telemetry_data = ta.stop_monitoring_online()

                ttnn.DumpDeviceProfiler(device)

                for i, v in telemetry_data.items():
                    data_info[i] += np.mean(v)

                inference_time_avg = profiler.get("run") / measure_iters
                data_info["tflops"] = 2 * m * k * n / 1e12 / inference_time_avg
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
                ideal_cycle_full_grid = m * k * n / 32 / 32 / 32 * cycle_per_tile / num_cores_full_grid
                ideal_cycle_user_grid = m * k * n / 32 / 32 / 32 * cycle_per_tile / num_cores_user_grid
                inference_cycle = inference_time_avg * get_device_freq() * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                utilization_full_grid_percentage = f"{utilization_full_grid * 100:.2f}%"
                utilization_user_grid_percentage = f"{utilization_user_grid * 100:.2f}%"

                data_info["run"] = inference_time_avg * 1e6
                data_info["utilization_vs_full_grid_perc"] = utilization_full_grid_percentage
                data_info["utilization_vs_user_grid_perc"] = utilization_user_grid_percentage

                log_infos = [f"{k}: {v}" for k, v in data_info.items()]
                logger.info(f"\nM={m} ==> \n {log_infos}")

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(data_info.values())
