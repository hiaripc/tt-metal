# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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


matmul_shapes_oob = [
    # (512, 512, 512),
    # (1024, 1024, 1024),
    # (2048, 2048, 2048),
    # (3072, 3072, 3072),
    (4096, 4096, 4096),
]

matmul_configs_oob = [
    (ttnn.bfloat16, False),
    (ttnn.bfloat16, True),
    (ttnn.bfloat8_b, False),
    (ttnn.bfloat8_b, True),
    (ttnn.bfloat4_b, False),
    (ttnn.bfloat4_b, True),
]


# @pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [(8, 8)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [100])
def test_matmul_2d_host_perf_out_of_box(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
    use_program_cache,
):
    FILE_NAME = "/home/bach/wd/nn/matmul/results/matmul2d_oob.csv"

    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    with open(FILE_NAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "m",
                "k",
                "n",
                "use_trace",
                "grid_size",
                "in0_sharded",
                "out_sharded",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "inference_time_avg",
                "trace_time",
                "offloading_time_in0",
                "offloading_time_in1",
                "TFLOPs (avg)",
                "Utilization (vs user grid)",
                "Utilization (vs 8x8 full grid)",
            ]
        )

        for dtype, use_trace in matmul_configs_oob:
            matmul_shapes = matmul_shapes_oob
            if dtype == ttnn.bfloat16:
                math_fidelity = ttnn.MathFidelity.HiFi2
            elif dtype == ttnn.bfloat8_b:
                math_fidelity = ttnn.MathFidelity.LoFi
            elif dtype == ttnn.bfloat4_b:
                math_fidelity = ttnn.MathFidelity.LoFi
            for m, k, n in matmul_shapes:
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                in0_storage_type = "DRAM"
                in1_storage_type = "DRAM"
                out_storage_type = "DRAM"

                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((tile_h, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, tile_w)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                output_t = in0_t @ in1_t

                for iter in range(0, num_warmup_iterations):
                    output_t = in0_t @ in1_t

                if use_trace:
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    for iter in range(0, num_measurement_iterations):
                        output_t = in0_t @ in1_t
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    profiler.start(f"run")
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")
                    ttnn.release_trace(device, tid)
                else:
                    profiler.start(f"run")
                    for iter in range(0, num_measurement_iterations):
                        output_t = in0_t @ in1_t
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")

                ttnn.DumpDeviceProfiler(device)

                inference_time_avg = profiler.get("run") / num_measurement_iterations
                tflops = 2 * m * k * n / 1e12 / inference_time_avg
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
                inference_cycle = inference_time_avg * get_device_freq() * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                utilization_full_grid_percentage = f"{utilization_full_grid * 100:.2f}%"
                utilization_user_grid_percentage = f"{utilization_user_grid * 100:.2f}%"
                logger.info(
                    f"M*K*N = {m}*{k}*{n} == inference time (avg): {inference_time_avg}, tflops (avg): {tflops}, utilization (vs user grid): {utilization_user_grid_percentage}, utilization (vs 8x8 grid): {utilization_full_grid_percentage}"
                )

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(
                    [
                        m,
                        k,
                        n,
                        f"{True}" if use_trace else f"{False}",
                        grid_size,
                        in0_storage_type,
                        in1_storage_type,
                        out_storage_type,
                        dtype,
                        math_fidelity,
                        f"{inference_time_avg * 1e6:.2f}",
                        f"{tflops:.2f}",
                        utilization_user_grid_percentage,
                        utilization_full_grid_percentage,
                    ]
                )
