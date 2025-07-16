// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);

    // All Gather specific
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc = get_arg_val<uint32_t>(rt_args_idx++);

    volatile tt_l1_ptr uint32_t* l1_signal_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);
    uint64_t remote_signal_semaphore_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, signal_semaphore_addr, noc);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles * in0_single_tile_size_bytes;

    // Push the local shard
    cb_reserve_back(cb_id_in0, batch * shard_size_in_tiles);
    cb_push_back(cb_id_in0, batch * shard_size_in_tiles);

    cb_reserve_back(cb_id_in2, batch * (ring_size - 1) * shard_size_in_tiles);

    uint32_t local_shard_read_addr = get_read_ptr(cb_id_in0);
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in2);

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t shard_cnt = 0; shard_cnt < ring_size; shard_cnt++) {
            uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * shard_cnt;
            uint64_t remote_curr_shard_write_addr =
                get_noc_addr(next_core_noc_x, next_core_noc_y, curr_shard_write_addr, noc);
            uint32_t curr_shard_read_addr =
                shard_cnt == 0 ? local_shard_read_addr : l1_write_addr_in0 + shard_size_bytes * (shard_cnt - 1);

            // Wait for signal from previous core that data has been added to this core's in0
            noc_semaphore_wait_min(l1_signal_sem_addr, shard_cnt);

            // Send data to next core
            if (shard_cnt < ring_size - 1) {  // Skip sending the last shard
                noc_async_write(curr_shard_read_addr, remote_curr_shard_write_addr, shard_size_bytes, noc);

                // Signal the next core that data is ready
                noc_semaphore_inc(remote_signal_semaphore_addr, 1, noc);
            }

            // Do stuff for matmul fusion here
            if (shard_cnt > 0) {
                cb_push_back(cb_id_in2, shard_size_in_tiles);
            }
        }
    }
}
