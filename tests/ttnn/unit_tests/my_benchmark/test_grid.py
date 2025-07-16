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


FILE_NAME_GRID = "/home/bach/wd/nn/matmul/results/grid_size.csv"

# tilziation on device only supports bf16
matmul_configs = [
    # ("f16_m2", ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
    ("f16_m4", ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
]

WARMUP_ITERS = 5
MEASURE_ITERS = 300

matmul_shapes_bfloat16 = [
    # (256, 256, 256),
    # (512, 512, 512),
    # (1024, 1024, 1024),
    # (2048, 2048, 2048),
    # (3072, 3072, 3072),
    # (4096, 4096, 4096),
    # (5120, 5120, 5120),
    # (6144, 6144, 6144),
    # (7168, 7168, 7168),
    (8192, 8192, 8192),
]

grid_core_map = {
    # "1": (1,1),
    # "2": (2,1),
    # "4": (2,2),
    # "8": (4,2),
    # "16": (4,4),
    # "32": (8,4),
    # "64": (8,8),
    "72": (8, 9),
    "80": (8, 10),
    "88": (8, 11),
    "96": (8, 12),
}


# launch with script to reset!
# ./run_test_grid.sh in TT_HOME
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("warmup_iters", [5])
@pytest.mark.parametrize("measure_iters", [300])
def test_grid_size(
    device,
    warmup_iters,
    measure_iters,
    use_program_cache,
):
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
    ]

    timing_infos = [
        "conversion_in0",
        "conversion_in1",
        "transfer_in0",
        "transfer_in1",
        "tilization_in0",
        "tilization_in1",
        "run",
    ]

    telemetry_infos = ["voltage", "current", "power", "aiclk", "temp"]

    for k in conf_infos + timing_infos + telemetry_infos:
        data_info[k] = None

    # Out of box, the tensors are always in DRAM
    data_info["in0_storage_type"] = "DRAM"
    data_info["in1_storage_type"] = "DRAM"
    data_info["out_storage_type"] = "DRAM"

    # for grid_x, grid_y in zip(range(8, 9), range(8,9)):

    # grid_s = os.getenv("GRID_SIZE", None)
    # assert grid_s, "set GRID_SIZE env!!"

    grid_s = "4"
    grid_x = grid_core_map[grid_s][0]
    grid_y = grid_core_map[grid_s][1]
    logger.info(f"\nSelecting grid size ({grid_x, grid_y}) ...")

    data_info["grid_size"] = (grid_x, grid_y)

    data_info["iters"] = measure_iters

    def timed_exec_add(timer: str, f):
        profiler.start(timer)
        res = f()
        ttnn.synchronize_device(device)
        profiler.end(timer)
        data_info[timer] += profiler.get(timer)
        return res

    if not os.path.exists(FILE_NAME_GRID):
        with open(FILE_NAME_GRID, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_info.keys())

    with open(FILE_NAME_GRID, mode="a", newline="") as file:
        writer = csv.writer(file)
        # header must be already setted
        # writer.writerow(data_info.keys())

        for conf, dtype, math_fidelity in matmul_configs:
            data_info["conf"] = conf
            data_info["dtype"] = dtype
            data_info["math_fidelity"] = math_fidelity

            logger.info(
                f"\n\nRunning conf {data_info['dtype']} ==> Type: {data_info['dtype']}, MF: {math_fidelity}\n\n"
            )
            if data_info["dtype"] == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16
            else:
                raise ValueError("Metti il dtype giusto!")

            for m, k, n in matmul_shapes:
                data_info["m"] = m
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                logger.info(f"\nM={m} \n")

                # set to 0
                for k in timing_infos:
                    data_info[k] = 0

                for k in telemetry_infos:
                    data_info[k] = 0

                # Warm up
                for i in range(warmup_iters):
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()
                    in0_t = ttnn.from_torch(
                        in0, tile=ttnn.Tile((32, 32)), dtype=data_info["dtype"], layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    in0_t = ttnn.to_device(in0_t, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    in0_t = ttnn.tilize(in0_t)

                    in1_t = ttnn.from_torch(
                        in1, tile=ttnn.Tile((32, 32)), dtype=data_info["dtype"], layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    in1_t = ttnn.to_device(in1_t, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    in1_t = ttnn.tilize(in1_t)

                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
                    )

                for i in range(measure_iters):
                    in0 = torch.ones(in0_shape).bfloat16()
                    in1 = torch.randn(in1_shape).bfloat16()
                    # in0
                    in0_t = timed_exec_add(
                        "conversion_in0",
                        lambda: ttnn.from_torch(
                            in0, tile=ttnn.Tile((32, 32)), dtype=data_info["dtype"], layout=ttnn.ROW_MAJOR_LAYOUT
                        ),
                    )

                    in0_t = timed_exec_add(
                        "transfer_in0",
                        lambda: ttnn.to_device(in0_t, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    )

                    in0_t = timed_exec_add("tilization_in0", lambda: ttnn.tilize(in0_t))

                    # in1
                    in1_t = timed_exec_add(
                        "conversion_in1",
                        lambda: ttnn.from_torch(
                            in1, tile=ttnn.Tile((32, 32)), dtype=data_info["dtype"], layout=ttnn.ROW_MAJOR_LAYOUT
                        ),
                    )

                    in1_t = timed_exec_add(
                        "transfer_in1",
                        lambda: ttnn.to_device(in1_t, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    )

                    in1_t = timed_exec_add("tilization_in1", lambda: ttnn.tilize(in1_t))

                    ta.start_monitoring_online(freq=1)
                    # run
                    output_t = timed_exec_add(
                        "inference_avg",
                        lambda: ttnn.matmul(
                            in0_t,
                            in1_t,
                            core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
                        ),
                    )
                    telemetry_data = ta.stop_monitoring_online()

                    for k, v in telemetry_data.items():
                        data_info[k] += np.mean(v)

                ttnn.DumpDeviceProfiler(device)

                for c in telemetry_infos:
                    data_info[c] /= measure_iters

                for c in timing_infos:
                    data_info[c] /= measure_iters
                    data_info[c] *= 1e6

                log_infos = [f"{k}: {v}" for k, v in data_info.items()]
                logger.info(f"M={m} ==> \n {log_infos}")

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
def test_grid_run_only(
    device,
    warmup_iters,
    measure_iters,
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
        "cores",
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

    data_info["iters"] = measure_iters

    if not os.path.exists(FILE_NAME_GRID):
        with open(FILE_NAME_GRID, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_info.keys())

    with open(FILE_NAME_GRID, mode="a", newline="") as file:
        writer = csv.writer(file)

        for cores, grid_size in grid_core_map.items():
            grid_x, grid_y = grid_size[0], grid_size[1]
            data_info["cores"] = cores
            data_info["grid_size"] = (grid_x, grid_y)
            logger.info(f"\nSelecting grid size ({grid_x, grid_y}) ...")

            for conf, dtype, math_fidelity in matmul_configs:
                data_info["conf"] = conf
                data_info["dtype"] = dtype
                data_info["math_fidelity"] = math_fidelity

                logger.info(
                    f"\n\nRunning conf {data_info['dtype']} ==> Type: {data_info['dtype']}, MF: {math_fidelity}\n\n"
                )

                for m, k, n in matmul_shapes_bfloat16:
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
                            core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
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
                            core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
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
                    num_cores_user_grid = grid_x * grid_y
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
