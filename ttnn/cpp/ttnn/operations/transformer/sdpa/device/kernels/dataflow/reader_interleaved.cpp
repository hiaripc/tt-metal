// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, volatile tt_l1_ptr const uint32_t* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;

    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = page_table_ptr[virtual_block];
    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + head_offset + block_offset;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t Skt = get_compile_time_arg_val(4);
    constexpr uint32_t DHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr uint32_t is_causal = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(13) == 1;
    constexpr uint32_t page_table_is_dram = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(15);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(16);

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunked_q_chunk_offset = get_arg_val<uint32_t>(argidx++);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_6;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr DataFormat mask_data_format = get_dataformat(cb_mask_in);

    constexpr uint32_t q_heads_per_kv = NQH / NKH;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr, .page_size = q_tile_bytes, .data_format = q_data_format};

    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr, .page_size = k_tile_bytes, .data_format = k_data_format};

    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr, .page_size = v_tile_bytes, .data_format = v_data_format};

    const InterleavedAddrGenFast<is_dram> mask_reader = {
        .bank_base_address = mask_addr, .page_size = mask_tile_bytes, .data_format = mask_data_format};

    volatile tt_l1_ptr uint32_t* page_table_ptr;

    uint32_t q_tile_id = 0;
    uint32_t k_tile_id = 0;
    uint32_t v_tile_id = 0;
    uint32_t mask_tile_id = 0;
    uint32_t barrier_count = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        if constexpr (is_chunked) {
            // Chunked means that we have paged attention
            const InterleavedAddrGen<page_table_is_dram> page_table_gen = {
                .bank_base_address = page_table_addr, .page_size = page_table_stick_size};
            cb_reserve_back(cb_id_page_table, 1);
            uint32_t page_table_cb_wr_ptr = get_write_ptr(cb_id_page_table);
            uint64_t page_table_noc_addr = get_noc_addr(nb, page_table_gen);
            noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_stick_size);
            noc_async_read_barrier();
            cb_push_back(cb_id_page_table, 1);
            page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);
        }
        const uint32_t q_batch_offset = nb * NQH * Sqt * DHt;
        const uint32_t kv_batch_offset = nb * NKH * Skt * DHt;
        const uint32_t mask_batch_offset = nb * Sqt * Skt;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
                uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                if (q_iter < q_chunk_div_2) {  // bottom half
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunk_div_2;  // Back half should start at 0
                    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                }
#else
                q_chunk = local_q_start + q_iter;
#endif

                uint32_t q_head_offset = nq * Sqt * DHt;
                uint32_t q_chunk_offset = q_chunk * Sq_chunk_t * DHt;
                q_tile_id = q_batch_offset + q_head_offset + q_chunk_offset;

                // Read Q chunk
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                uint32_t q_write_ptr = get_write_ptr(cb_q_in);

                barrier_count = 0;
                for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
                    noc_async_read_tile(q_tile_id, q_reader, q_write_ptr);
                    q_tile_id += 1;
                    q_write_ptr += q_tile_bytes;

                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                }
                noc_async_read_barrier();

                cb_push_back(cb_q_in, q_chunk_tiles);

                if constexpr (is_chunked) {
                    q_chunk = chunked_q_chunk_offset + q_chunk;
                }
                uint32_t q_low_idx =
                    q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
                uint32_t q_high_idx;
                if constexpr (is_causal) {
                    q_high_idx = q_low_idx + Sq_chunk_t;
                } else {
                    q_high_idx = Skt;
                }

                const uint32_t kv_head = nq / q_heads_per_kv;
                const uint32_t kv_head_offset = kv_head * Skt * DHt;

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                    const uint32_t k_start_tile_id = kv_batch_offset + kv_head_offset + k_chunk * Sk_chunk_t * DHt;

                    if constexpr (is_chunked) {
                        // Use page table to read K chunk
                        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        cb_reserve_back(cb_k_in, k_chunk_tiles);
                        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
                        barrier_count = 0;
                        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                            uint32_t k_write_ptr_col = k_write_ptr + row * k_tile_bytes;
                            uint32_t virtual_k_tile_row_num = k_chunk_start_row_num + row;
                            uint32_t physical_k_tile_id =
                                virtual_seq_tile_id_to_physical_tile_id<NKH, block_size_t, DHt>(
                                    virtual_k_tile_row_num, kv_head, page_table_ptr);
                            for (uint32_t col = 0; col < DHt; ++col) {
                                noc_async_read_tile(physical_k_tile_id, k_reader, k_write_ptr_col);
                                physical_k_tile_id += 1;                       // Go to next tile in row
                                k_write_ptr_col += Sk_chunk_t * k_tile_bytes;  // Go to next column in CB

                                if (++barrier_count == barrier_threshold) {
                                    noc_async_read_barrier();
                                    barrier_count = 0;
                                }
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_k_in, k_chunk_tiles);

                    } else {
                        // Read K chunk transposed
                        cb_reserve_back(cb_k_in, k_chunk_tiles);
                        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
                        barrier_count = 0;
                        for (uint32_t col = 0; col < DHt; ++col) {
                            k_tile_id = k_start_tile_id + col;
                            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                                noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                                k_tile_id += DHt;
                                k_write_ptr += k_tile_bytes;

                                if (++barrier_count == barrier_threshold) {
                                    noc_async_read_barrier();
                                    barrier_count = 0;
                                }
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_k_in, k_chunk_tiles);
                    }

                    if constexpr (use_provided_mask) {
                        // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                        // Q-range = [q_low, q_high)
                        // K-range = [k_low, k_high)
                        // does_overlap = not (q_low >= k_high or k_low >= q_high)
                        // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                        // Read mask chunk
                        cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                        uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                        barrier_count = 0;
                        mask_tile_id = mask_batch_offset + q_chunk * Sq_chunk_t * Skt /*row_offset*/ + k_chunk * Sk_chunk_t /*col_offset*/;
                        for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                            for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                                noc_async_read_tile(mask_tile_id, mask_reader, mask_write_ptr);
                                mask_tile_id += 1;
                                mask_write_ptr += mask_tile_bytes;
                                if (++barrier_count == barrier_threshold) {
                                    noc_async_read_barrier();
                                    barrier_count = 0;
                                }
                            }
                            // Strid along columns to get to next row
                            mask_tile_id -= Sk_chunk_t;
                            mask_tile_id += Skt;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_mask_in, mask_chunk_tiles);
                    }

                    if constexpr (is_chunked) {
                        // Use page table to read V chunk
                        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        cb_reserve_back(cb_v_in, k_chunk_tiles);
                        uint32_t v_write_ptr = get_write_ptr(cb_v_in);
                        barrier_count = 0;

                        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                            uint32_t virtual_v_tile_row_num = k_chunk_start_row_num + row;
                            uint32_t physical_v_tile_id =
                                virtual_seq_tile_id_to_physical_tile_id<NKH, block_size_t, DHt>(
                                    virtual_v_tile_row_num, kv_head, page_table_ptr);
                            for (uint32_t col = 0; col < DHt; ++col) {
                                noc_async_read_tile(physical_v_tile_id, v_reader, v_write_ptr);
                                physical_v_tile_id += 1;
                                v_write_ptr += v_tile_bytes;

                                if (++barrier_count == barrier_threshold) {
                                    noc_async_read_barrier();
                                    barrier_count = 0;
                                }
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_v_in, k_chunk_tiles);
                    } else {
                        v_tile_id = k_start_tile_id;
                        // Read V chunk
                        cb_reserve_back(cb_v_in, k_chunk_tiles);
                        uint32_t v_write_ptr = get_write_ptr(cb_v_in);
                        barrier_count = 0;
                        for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
                            noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
                            v_tile_id += 1;
                            v_write_ptr += v_tile_bytes;

                            if (++barrier_count == barrier_threshold) {
                                noc_async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_v_in, k_chunk_tiles);
                    }
                }
            }
        }

        if constexpr (is_chunked) {
            cb_pop_front(cb_id_page_table, 1);
        }
    }
}
