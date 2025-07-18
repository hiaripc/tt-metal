// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "dev_msgs.h"
#include "device/device_handle.hpp"
#include "llrt/hal.hpp"
#include "program_command_sequence.hpp"
#include "tt_metal/command_queue.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hw/inc/circular_buffer_constants.h"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/impl/event/event.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program/program_dispatch_utils.hpp"
#include "umd/device/tt_xy_pair.h"

#include "llrt/hal.hpp"

using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace detail {

bool DispatchStateCheck(bool isFastDispatch) {
    static bool fd = isFastDispatch;
    TT_FATAL(fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!");
    return fd;
}

void SetLazyCommandQueueMode(bool lazy) {
    DispatchStateCheck(true);
    LAZY_COMMAND_QUEUE_MODE = lazy;
}
}  // namespace detail

enum DispatchWriteOffsets {
    DISPATCH_WRITE_OFFSET_ZERO = 0,
    DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE = 1,
    DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE = 2,
};

// EnqueueReadBufferCommandSection

EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    Buffer& buffer,
    void* dst,
    SystemMemoryManager& manager,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    dst(dst),
    manager(manager),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    sub_device_ids(sub_device_ids),
    src_page_index(src_page_index),
    pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {
    TT_ASSERT(buffer.is_dram() or buffer.is_l1(), "Trying to read an invalid buffer");

    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void EnqueueReadInterleavedBufferCommand::add_prefetch_relay(HugepageDeviceCommand& command) {
    uint32_t padded_page_size = this->buffer.aligned_page_size();
    command.add_prefetch_relay_paged(
        this->buffer.is_dram(), this->src_page_index, this->buffer.address(), padded_page_size, this->pages_to_read);
}

void EnqueueReadShardedBufferCommand::add_prefetch_relay(HugepageDeviceCommand& command) {
    uint32_t padded_page_size = this->buffer.aligned_page_size();
    const CoreCoord virtual_core =
        this->buffer.device()->virtual_core_from_logical_core(this->core, this->buffer.core_type());
    command.add_prefetch_relay_linear(
        this->device->get_noc_unicast_encoding(this->noc_index, virtual_core),
        padded_page_size * this->pages_to_read,
        this->bank_base_address);
}

void EnqueueReadBufferCommand::process() {
    uint32_t num_worker_counters = this->sub_device_ids.size();
    // accounts for padding
    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) * num_worker_counters +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        hal.get_alignment(HalMemType::HOST) +  // CQ_PREFETCH_CMD_STALL
        hal.get_alignment(HalMemType::HOST) +  // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        hal.get_alignment(HalMemType::HOST);   // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    uint32_t dispatch_message_base_addr = dispatch_constants::get(dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier + prefetch stall for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = this->sub_device_ids[i].to_index();
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, this->expected_num_workers_completed[offset_index ]);

    }
    auto offset_index = this->sub_device_ids[last_index].to_index();
    uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
    command_sequence.add_dispatch_wait_with_prefetch_stall(
        true, dispatch_message_addr, this->expected_num_workers_completed[offset_index]);

    uint32_t padded_page_size = this->buffer.aligned_page_size();
    bool flush_prefetch = false;
    command_sequence.add_dispatch_write_host(flush_prefetch, this->pages_to_read * padded_page_size, false);

    this->add_prefetch_relay(command_sequence);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

// EnqueueWriteBufferCommand section

EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    const Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    bool issue_wait,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    uint32_t bank_base_address,
    uint32_t padded_page_size,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    manager(manager),
    issue_wait(issue_wait),
    src(src),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    sub_device_ids(sub_device_ids),
    bank_base_address(bank_base_address),
    padded_page_size(padded_page_size),
    dst_page_index(dst_page_index),
    pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(buffer.is_dram() or buffer.is_l1(), "Trying to write to an invalid buffer");
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void EnqueueWriteInterleavedBufferCommand::add_dispatch_write(HugepageDeviceCommand& command_sequence) {
    uint8_t is_dram = uint8_t(this->buffer.is_dram());
    TT_ASSERT(
        this->dst_page_index <= 0xFFFF,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    uint16_t start_page = uint16_t(this->dst_page_index & 0xFFFF);
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch, is_dram, start_page, this->bank_base_address, this->padded_page_size, this->pages_to_write);
}

void EnqueueWriteInterleavedBufferCommand::add_buffer_data(HugepageDeviceCommand& command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;

    uint32_t full_page_size = this->buffer.aligned_page_size();  // this->padded_page_size could be a partial page if
                                                                 // buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = this->padded_page_size < full_page_size;

    uint32_t buffer_addr_offset = this->bank_base_address - this->buffer.address();
    uint32_t num_banks = this->device->num_banks(this->buffer.buffer_type());

    // TODO: Consolidate
    if (write_partial_pages) {
        uint32_t padding = full_page_size - this->buffer.page_size();
        uint32_t unpadded_src_offset = buffer_addr_offset;
        uint32_t src_address_offset = unpadded_src_offset;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
             sysmem_address_offset += this->padded_page_size) {
            uint32_t page_size_to_copy = this->padded_page_size;
            if (src_address_offset + this->padded_page_size > buffer.page_size()) {
                // last partial page being copied from unpadded src buffer
                page_size_to_copy -= padding;
            }
            command_sequence.add_data((char*)this->src + src_address_offset, page_size_to_copy, this->padded_page_size);
            src_address_offset += page_size_to_copy;
        }
    } else {
        uint32_t unpadded_src_offset =
            (((buffer_addr_offset / this->padded_page_size) * num_banks) + this->dst_page_index) *
            this->buffer.page_size();
        if (this->buffer.page_size() % this->buffer.alignment() != 0 and
            this->buffer.page_size() != this->buffer.size()) {
            // If page size is not aligned, we cannot do a contiguous write
            uint32_t src_address_offset = unpadded_src_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                 sysmem_address_offset += this->padded_page_size) {
                command_sequence.add_data(
                    (char*)this->src + src_address_offset, this->buffer.page_size(), this->padded_page_size);
                src_address_offset += this->buffer.page_size();
            }
        } else {
            command_sequence.add_data((char*)this->src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void EnqueueWriteShardedBufferCommand::add_dispatch_write(HugepageDeviceCommand& command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    const CoreCoord virtual_core =
        this->buffer.device()->virtual_core_from_logical_core(this->core, this->buffer.core_type());
    command_sequence.add_dispatch_write_linear(
        0,
        this->device->get_noc_unicast_encoding(this->noc_index, virtual_core),
        this->bank_base_address,
        data_size_bytes);
}

void EnqueueWriteShardedBufferCommand::add_buffer_data(HugepageDeviceCommand& command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    if (this->buffer_page_mapping) {
        const auto& page_mapping = *this->buffer_page_mapping;
        uint8_t* dst = command_sequence.reserve_space<uint8_t*, true>(data_size_bytes);
        // TODO: Expose getter for cmd_write_offsetB?
        uint32_t dst_offset = dst - (uint8_t*)command_sequence.data();
        for (uint32_t dev_page = this->dst_page_index; dev_page < this->dst_page_index + this->pages_to_write;
             ++dev_page) {
            auto& host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                command_sequence.update_cmd_sequence(
                    dst_offset,
                    (char*)this->src + host_page.value() * this->buffer.page_size(),
                    this->buffer.page_size());
            }
            dst_offset += this->padded_page_size;
        }
    } else {
        if (this->buffer.page_size() != this->padded_page_size and this->buffer.page_size() != this->buffer.size()) {
            uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
            for (uint32_t i = 0; i < this->pages_to_write; ++i) {
                command_sequence.add_data(
                    (char*)this->src + unpadded_src_offset, this->buffer.page_size(), this->padded_page_size);
                unpadded_src_offset += this->buffer.page_size();
            }
        } else {
            uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
            command_sequence.add_data((char*)this->src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void EnqueueWriteBufferCommand::process() {
    uint32_t num_worker_counters = this->sub_device_ids.size();
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(
        sizeof(CQPrefetchCmd) + // CQ_PREFETCH_CMD_RELAY_INLINE
        sizeof(CQDispatchCmd) + // CQ_DISPATCH_CMD_WRITE_PAGED or CQ_DISPATCH_CMD_WRITE_LINEAR
        data_size_bytes, pcie_alignment);
    if (this->issue_wait) {
        cmd_sequence_sizeB += hal.get_alignment(HalMemType::HOST) * num_worker_counters;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (this->issue_wait) {
        uint32_t dispatch_message_base_addr = dispatch_constants::get(
            this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
        for (const auto &sub_device_id : this->sub_device_ids) {
            auto offset_index = sub_device_id.to_index();
            uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
            command_sequence.add_dispatch_wait(false, dispatch_message_addr, this->expected_num_workers_completed[offset_index]);
        }
    }

    this->add_dispatch_write(command_sequence);

    uint32_t full_page_size = this->buffer.aligned_page_size();  // this->padded_page_size could be a partial page if
                                                                 // buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = this->padded_page_size < full_page_size;

    this->add_buffer_data(command_sequence);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

inline uint32_t get_packed_write_max_unicast_sub_cmds(Device* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}

// EnqueueProgramCommand Section

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    Program& program,
    CoreCoord& dispatch_core,
    SystemMemoryManager& manager,
    WorkerConfigBufferMgr& config_buffer_mgr,
    uint32_t expected_num_workers_completed,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    SubDeviceId sub_device_id) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    manager(manager),
    config_buffer_mgr(config_buffer_mgr),
    expected_num_workers_completed(expected_num_workers_completed),
    program(program),
    dispatch_core(dispatch_core),
    multicast_cores_launch_message_wptr(multicast_cores_launch_message_wptr),
    unicast_cores_launch_message_wptr(unicast_cores_launch_message_wptr),
    sub_device_id(sub_device_id) {
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    this->packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
    this->dispatch_message_addr = dispatch_constants::get(
        this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE) +
        dispatch_constants::get(this->dispatch_core_type).get_dispatch_message_offset(this->sub_device_id.to_index());
}

void EnqueueProgramCommand::write_program_command_sequence(
    const ProgramCommandSequence& program_command_sequence, bool stall_first, bool stall_before_program) {
    TT_ASSERT(!(stall_first && stall_before_program));
    uint32_t preamble_fetch_size_bytes = program_command_sequence.preamble_command_sequence.size_bytes();
    auto& curr_stall_seq_idx = program_command_sequence.current_stall_seq_idx;
    uint32_t stall_fetch_size_bytes =
        (stall_first || stall_before_program) ? program_command_sequence.stall_command_sequences[curr_stall_seq_idx].size_bytes() : 0;

    uint32_t runtime_args_fetch_size_bytes = program_command_sequence.runtime_args_fetch_size_bytes;

    uint32_t program_fetch_size_bytes = program_command_sequence.device_command_sequence.size_bytes();

    uint32_t program_config_buffer_data_size_bytes =
        program_command_sequence.program_config_buffer_data_size_bytes;

    uint32_t program_rem_fetch_size_bytes = program_fetch_size_bytes - program_config_buffer_data_size_bytes;

    uint8_t* program_command_sequence_data = (uint8_t*)program_command_sequence.device_command_sequence.data();

    uint32_t total_fetch_size_bytes =
        stall_fetch_size_bytes + preamble_fetch_size_bytes + runtime_args_fetch_size_bytes + program_fetch_size_bytes;

    if (total_fetch_size_bytes <= dispatch_constants::get(this->dispatch_core_type).max_prefetch_command_size()) {
        this->manager.issue_queue_reserve(total_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);

        this->manager.cq_write(
            program_command_sequence.preamble_command_sequence.data(), preamble_fetch_size_bytes, write_ptr);
        write_ptr += preamble_fetch_size_bytes;

        if (stall_first) {
            // Must stall before writing runtime args
            this->manager.cq_write(
                program_command_sequence.stall_command_sequences[curr_stall_seq_idx].data(), stall_fetch_size_bytes, write_ptr);
            write_ptr += stall_fetch_size_bytes;
        }

        for (const auto& cmds : program_command_sequence.runtime_args_command_sequences) {
            this->manager.cq_write(cmds.data(), cmds.size_bytes(), write_ptr);
            write_ptr += cmds.size_bytes();
        }

        if (stall_before_program) {
            if (program_config_buffer_data_size_bytes > 0) {
                this->manager.cq_write(program_command_sequence_data, program_config_buffer_data_size_bytes, write_ptr);
                program_command_sequence_data += program_config_buffer_data_size_bytes;
                write_ptr += program_config_buffer_data_size_bytes;
            }

            // Didn't stall before kernel config data, stall before remaining commands
            this->manager.cq_write(
                program_command_sequence.stall_command_sequences[curr_stall_seq_idx].data(), stall_fetch_size_bytes, write_ptr);
            write_ptr += stall_fetch_size_bytes;

            this->manager.cq_write(program_command_sequence_data, program_rem_fetch_size_bytes, write_ptr);
        } else {
            this->manager.cq_write(program_command_sequence_data, program_fetch_size_bytes, write_ptr);
        }

        this->manager.issue_queue_push_back(total_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for entire program
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(total_fetch_size_bytes, this->command_queue_id);

        // TODO: We are making a lot of fetch queue entries here, we can pack multiple commands into one fetch q entry
    } else {
        this->manager.issue_queue_reserve(preamble_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(
            program_command_sequence.preamble_command_sequence.data(), preamble_fetch_size_bytes, write_ptr);
        this->manager.issue_queue_push_back(preamble_fetch_size_bytes, this->command_queue_id);
        // One fetch queue entry for just the wait and stall, very inefficient
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(preamble_fetch_size_bytes, this->command_queue_id);

        if (stall_first) {
            // Must stall before writing kernel config data
            this->manager.issue_queue_reserve(stall_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(
                program_command_sequence.stall_command_sequences[curr_stall_seq_idx].data(), stall_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(stall_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for just the wait and stall, very inefficient
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(stall_fetch_size_bytes, this->command_queue_id);
        }

        // TODO: We can pack multiple RT args into one fetch q entry
        for (const auto& cmds : program_command_sequence.runtime_args_command_sequences) {
            uint32_t fetch_size_bytes = cmds.size_bytes();
            this->manager.issue_queue_reserve(fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(cmds.data(), fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for each runtime args write location, e.g. BRISC/NCRISC/TRISC/ERISC
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
        }

        // Insert a stall between program data that goes on the ring buffer and the rest of the data
        // Otherwise write all data in 1 prefetch entry
        if (stall_before_program) {
            if (program_config_buffer_data_size_bytes > 0) {
                this->manager.issue_queue_reserve(program_config_buffer_data_size_bytes, this->command_queue_id);
                write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
                this->manager.cq_write(program_command_sequence_data, program_config_buffer_data_size_bytes, write_ptr);
                this->manager.issue_queue_push_back(program_config_buffer_data_size_bytes, this->command_queue_id);
                this->manager.fetch_queue_reserve_back(this->command_queue_id);
                this->manager.fetch_queue_write(program_config_buffer_data_size_bytes, this->command_queue_id);
                program_command_sequence_data += program_config_buffer_data_size_bytes;
            }

            // Didn't stall before kernel config data, stall before remaining commands
            this->manager.issue_queue_reserve(stall_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(
                program_command_sequence.stall_command_sequences[curr_stall_seq_idx].data(), stall_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(stall_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for just the wait and stall, very inefficient
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(stall_fetch_size_bytes, this->command_queue_id);

            this->manager.issue_queue_reserve(program_rem_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(program_command_sequence_data, program_rem_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(program_rem_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for rest of program commands
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(program_rem_fetch_size_bytes, this->command_queue_id);
        } else {
            this->manager.issue_queue_reserve(program_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(program_command_sequence_data, program_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(program_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for rest of program commands
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(program_fetch_size_bytes, this->command_queue_id);
        }
    }
}

void EnqueueProgramCommand::process() {
    // Dispatch metadata contains runtime information based on
    // the kernel config ring buffer state
    program_utils::ProgramDispatchMetadata dispatch_metadata;

    // Compute the total number of workers this program uses
    uint32_t num_workers = 0;
    if (program.runs_on_noc_multicast_only_cores()) {
        num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, this->sub_device_id);
    }
    if (program.runs_on_noc_unicast_only_cores()) {
        num_workers += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, this->sub_device_id);
    }
    // Reserve space for this program in the kernel config ring buffer
    program_utils::reserve_space_in_kernel_config_buffer(
        this->config_buffer_mgr,
        program.get_program_config_sizes(),
        program.get_program_binary_status(device->id()),
        num_workers,
        this->expected_num_workers_completed,
        dispatch_metadata);

    // Remove launch buffers from config addrs, since they're not real cores.
    const tt::stl::Span<ConfigBufferEntry> kernel_config_addrs{
        dispatch_metadata.kernel_config_addrs.data(), dispatch_metadata.kernel_config_addrs.size() - 2};

    RecordProgramRun(program);

    // Access the program dispatch-command cache
    auto& cached_program_command_sequence = program.get_cached_program_command_sequences().begin()->second;
    // Update the generated dispatch commands based on the state of the CQ and the ring buffer
    program_utils::update_program_dispatch_commands(
        program,
        cached_program_command_sequence,
        kernel_config_addrs,
        this->multicast_cores_launch_message_wptr,
        this->unicast_cores_launch_message_wptr,
        this->expected_num_workers_completed,
        this->dispatch_core,
        this->dispatch_core_type,
        this->sub_device_id,
        dispatch_metadata,
        program.get_program_binary_status(device->id()));
    // Issue dispatch commands for this program
    this->write_program_command_sequence(cached_program_command_sequence, dispatch_metadata.stall_first, dispatch_metadata.stall_before_program);
    // Kernel Binaries are committed to DRAM, the first time the program runs on device. Reflect this on host.
    program.set_program_binary_status(device->id(), ProgramBinaryStatus::Committed);
}

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    SystemMemoryManager& manager,
    uint32_t event_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    bool clear_count,
    bool write_barrier) :
    command_queue_id(command_queue_id),
    device(device),
    noc_index(noc_index),
    manager(manager),
    event_id(event_id),
    expected_num_workers_completed(expected_num_workers_completed),
    sub_device_ids(sub_device_ids),
    clear_count(clear_count),
    write_barrier(write_barrier) {}

void EnqueueRecordEventCommand::process() {
    std::vector<uint32_t> event_payload(dispatch_constants::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = this->event_id;

    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    uint8_t num_hw_cqs =
        this->device->num_hw_cqs();  // Device initialize asserts that there can only be a maximum of 2 HW CQs
    uint32_t packed_event_payload_sizeB =
        align(sizeof(CQDispatchCmd) + num_hw_cqs * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment) +
        (align(dispatch_constants::EVENT_PADDED_SIZE, l1_alignment) * num_hw_cqs);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, pcie_alignment);
    uint32_t num_worker_counters = this->sub_device_ids.size();

    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) * num_worker_counters +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED + unicast subcmds + event
                              // payload
        align(
            sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
            pcie_alignment);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint32_t dispatch_message_base_addr = dispatch_constants::get(
        dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);

    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = this->sub_device_ids[i].to_index();
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, this->expected_num_workers_completed[offset_index], this->clear_count);

    }
    auto offset_index = this->sub_device_ids[last_index].to_index();
    uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
    command_sequence.add_dispatch_wait(
            this->write_barrier, dispatch_message_addr, this->expected_num_workers_completed[offset_index], this->clear_count);

    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_hw_cqs);
    std::vector<std::pair<const void*, uint32_t>> event_payloads(num_hw_cqs);

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair dispatch_location;
        if (device->is_mmio_capable()) {
            dispatch_location = dispatch_core_manager::instance().dispatcher_core(this->device->id(), channel, cq_id);
        } else {
            dispatch_location = dispatch_core_manager::instance().dispatcher_d_core(this->device->id(), channel, cq_id);
        }

        CoreCoord dispatch_virtual_core = this->device->virtual_core_from_logical_core(dispatch_location, core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{
            .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, dispatch_virtual_core)};
        event_payloads[cq_id] = {event_payload.data(), event_payload.size() * sizeof(uint32_t)};
    }

    uint32_t completion_q0_last_event_addr = dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
    uint32_t address = this->command_queue_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;
    const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_hw_cqs,
        address,
        dispatch_constants::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads,
        packed_write_max_unicast_sub_cmds);

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_host<true>(
        flush_prefetch, dispatch_constants::EVENT_PADDED_SIZE, true, event_payload.data());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id,
    Device* device,
    SystemMemoryManager& manager,
    const Event& sync_event,
    bool clear_count) :
    command_queue_id(command_queue_id),
    device(device),
    manager(manager),
    sync_event(sync_event),
    clear_count(clear_count) {
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
    // TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
    //     "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
    //     event, command_queue_id);
}

void EnqueueWaitForEventCommand::process() {
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    uint32_t completion_q0_last_event_addr = dispatch_constants::get(this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
    uint32_t completion_q1_last_event_addr = dispatch_constants::get(this->dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);

    uint32_t last_completed_event_address =
        sync_event.cq_id == 0 ? completion_q0_last_event_addr : completion_q1_last_event_addr;

    command_sequence.add_dispatch_wait(false, last_completed_event_address, sync_event.event_id, this->clear_count);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

EnqueueTraceCommand::EnqueueTraceCommand(
    uint32_t command_queue_id,
    Device* device,
    SystemMemoryManager& manager,
    std::shared_ptr<detail::TraceDescriptor>& descriptor,
    Buffer& buffer,
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> & expected_num_workers_completed,
    NOC noc_index,
    CoreCoord dispatch_core) :
    command_queue_id(command_queue_id),
    buffer(buffer),
    device(device),
    manager(manager),
    descriptor(descriptor),
    expected_num_workers_completed(expected_num_workers_completed),
    clear_count(true),
    noc_index(noc_index),
    dispatch_core(dispatch_core) {}

void EnqueueTraceCommand::process() {
    uint32_t num_sub_devices = descriptor->descriptors.size();
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t go_signals_cmd_size = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * descriptor->descriptors.size();

    uint32_t cmd_sequence_sizeB =
        this->device->dispatch_s_enabled() * hal.get_alignment(HalMemType::HOST) + // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +  // go signal cmd
        (hal.get_alignment(HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
        // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s. dispatch_s is responsible for resetting worker count
        // and giving dispatch_d the latest worker state. This is encapsulated in the dispatch_s wait command (only to be sent when dispatch is distributed
        // on 2 cores)
        (this->device->distributed_dispatcher()) * hal.get_alignment(HalMemType::HOST)) * num_sub_devices +
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_EXEC_BUF

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (this->device->dispatch_s_enabled()) {
        uint16_t index_bitmask = 0;
        for (const auto &id : descriptor->sub_device_ids) {
            index_bitmask |= 1 << id.to_index();
        }
        command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
    }
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t dispatch_message_base_addr = dispatch_constants::get(
        dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    go_msg_t reset_launch_message_read_ptr_go_signal;
    reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
    reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)this->dispatch_core.x;
    reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)this->dispatch_core.y;
    for (const auto& [id, desc] : descriptor->descriptors) {
        const auto& noc_data_start_idx = device->noc_data_start_index(id, desc.num_traced_programs_needing_go_signal_multicast, desc.num_traced_programs_needing_go_signal_unicast);
        const auto& num_noc_mcast_txns = desc.num_traced_programs_needing_go_signal_multicast ? device->num_noc_mcast_txns(id) : 0;
        const auto& num_noc_unicast_txns = desc.num_traced_programs_needing_go_signal_unicast ? device->num_noc_unicast_txns(id) : 0;
        reset_launch_message_read_ptr_go_signal.dispatch_message_offset = (uint8_t)dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(id.to_index());
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(id.to_index());
        auto index = id.to_index();
        // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
        command_sequence.add_dispatch_go_signal_mcast(
            this->expected_num_workers_completed[index],
            *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal),
            dispatch_message_addr,
            num_noc_mcast_txns,
            num_noc_unicast_txns,
            noc_data_start_idx,
            dispatcher_for_go_signal);
        if (desc.num_traced_programs_needing_go_signal_multicast) {
            this->expected_num_workers_completed[index] += device->num_worker_cores(HalProgrammableCoreType::TENSIX, id);
        }
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            this->expected_num_workers_completed[index] += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, id);
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed this step, before sending kernel config data to workers
    // or notifying dispatch_s that its safe to send the go_signal.
    // Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (const auto &id : descriptor->sub_device_ids) {
        auto index = id.to_index();
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(index);
        if (this->device->distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, this->expected_num_workers_completed[index], this->clear_count, false, true, 1);
        }
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, this->expected_num_workers_completed[index], this->clear_count);
        if (this->clear_count) {
            this->expected_num_workers_completed[index] = 0;
        }
    }

    uint32_t page_size = buffer.page_size();
    uint32_t page_size_log2 = __builtin_ctz(page_size);
    TT_ASSERT((page_size & (page_size - 1)) == 0, "Page size must be a power of 2");

    command_sequence.add_prefetch_exec_buf(buffer.address(), page_size_log2, buffer.num_pages());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    const bool stall_prefetcher = true;
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id, stall_prefetcher);
}

EnqueueTerminateCommand::EnqueueTerminateCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager) :
    command_queue_id(command_queue_id), device(device), manager(manager) {}

void EnqueueTerminateCommand::process() {
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand dispatch_d_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_d_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_MASTER);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
    if (this->device->dispatch_s_enabled()) {
        // Terminate dispatch_s if enabled
        cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
        HugepageDeviceCommand dispatch_s_command_sequence(cmd_region, cmd_sequence_sizeB);
        dispatch_s_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_SLAVE);
        this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
    }
    cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

// HWCommandQueue section
HWCommandQueue::HWCommandQueue(Device* device, uint32_t id, NOC noc_index) :
    manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->noc_index = noc_index;
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();
    if (tt::Cluster::instance().is_galaxy_cluster()) {
        // Galaxy puts 4 devices per host channel until umd can provide one channel per device.
        this->size_B = this->size_B / 4;
    }

    CoreCoord enqueue_program_dispatch_core;
    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    if (this->device->num_hw_cqs() == 1 or core_type == CoreType::WORKER) {
        // dispatch_s exists with this configuration. Workers write to dispatch_s
        enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_s_core(device->id(), channel, id);
    }
    else {
        if (device->is_mmio_capable()) {
            enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_core(device->id(), channel, id);
        } else {
            enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_d_core(device->id(), channel, id);
        }
    }
    this->virtual_enqueue_program_dispatch_core =
        device->virtual_core_from_logical_core(enqueue_program_dispatch_core, core_type);

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::instance().completion_queue_writer_core(device->id(), channel, this->id);

    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    // Set the affinity of the completion queue reader.
    set_device_thread_affinity(this->completion_queue_thread, device->get_completion_queue_reader_core());

    for (uint32_t i = 0; i < dispatch_constants::DISPATCH_MESSAGE_ENTRIES; i++) {
        this->expected_num_workers_completed[i] = 0;
    }
    reset_config_buffer_mgr(dispatch_constants::DISPATCH_MESSAGE_ENTRIES);
}

void HWCommandQueue::set_num_worker_sems_on_dispatch(uint32_t num_worker_sems) {
    // Not needed for regular dispatch kernel
    if (!this->device->dispatch_s_enabled()) {
        return;
    }
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    command_sequence.add_dispatch_set_num_worker_sems(num_worker_sems, DispatcherSelect::DISPATCH_SLAVE);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);
}

void HWCommandQueue::set_go_signal_noc_data_on_dispatch(const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) {
    uint32_t pci_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + go_signal_noc_data.size() * sizeof(uint32_t), pci_alignment);
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    DispatcherSelect dispatcher_for_go_signal = this->device->dispatch_s_enabled() ? DispatcherSelect::DISPATCH_SLAVE : DispatcherSelect::DISPATCH_MASTER;
    command_sequence.add_dispatch_set_go_signal_noc_data(go_signal_noc_data, dispatcher_for_go_signal);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);
}

void HWCommandQueue::reset_worker_state(bool reset_launch_msg_state) {
    auto num_sub_devices = device->num_sub_devices();
    uint32_t go_signals_cmd_size = 0;
    if (reset_launch_msg_state) {
        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        go_signals_cmd_size = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * num_sub_devices;
    }
    uint32_t cmd_sequence_sizeB =
        reset_launch_msg_state * this->device->dispatch_s_enabled() * hal.get_alignment(HalMemType::HOST) + // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +  // go signal cmd
        (hal.get_alignment(HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
        // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s. dispatch_s is responsible for resetting worker count
        // and giving dispatch_d the latest worker state. This is encapsulated in the dispatch_s wait command (only to be sent when dispatch is distributed
        // on 2 cores)
        this->device->distributed_dispatcher() * hal.get_alignment(HalMemType::HOST)) * num_sub_devices;
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    bool clear_count = true;
    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t dispatch_message_base_addr = dispatch_constants::get(
        dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    if (reset_launch_msg_state) {
        if (device->dispatch_s_enabled()) {
            uint16_t index_bitmask = 0;
            for (uint32_t i = 0; i < num_sub_devices; ++i) {
                index_bitmask |= 1 << i;
            }
            command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
            dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
        }
        go_msg_t reset_launch_message_read_ptr_go_signal;
        reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
        reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)this->virtual_enqueue_program_dispatch_core.x;
        reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)this->virtual_enqueue_program_dispatch_core.y;
        for (uint32_t i = 0; i < num_sub_devices; ++i) {
            reset_launch_message_read_ptr_go_signal.dispatch_message_offset = (uint8_t)dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
            uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
            // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
            command_sequence.add_dispatch_go_signal_mcast(expected_num_workers_completed[i], *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal), dispatch_message_addr, device->num_noc_mcast_txns({i}), device->num_noc_unicast_txns({i}), device->noc_data_start_index({i}), dispatcher_for_go_signal);
            expected_num_workers_completed[i] += device->num_worker_cores(HalProgrammableCoreType::TENSIX, {i});
            expected_num_workers_completed[i] += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, {i});
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed this step, before sending kernel config data to workers
    // or notifying dispatch_s that its safe to send the go_signal.
    // Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
        if (device->distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, expected_num_workers_completed[i], clear_count, false, true, 1);
        }
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, expected_num_workers_completed[i], clear_count);
    }
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);

    if (clear_count) {
        std::fill(expected_num_workers_completed.begin(), expected_num_workers_completed.begin() + num_sub_devices, 0);
    }
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {
        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted "
            "commands: {}",
            this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->set_exit_condition();
        this->completion_queue_thread.join();
    }
}

void HWCommandQueue::increment_num_entries_in_completion_q() {
    // Increment num_entries_in_completion_q and inform reader thread
    // that there is work in the completion queue to process
    this->num_entries_in_completion_q++;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

void HWCommandQueue::set_exit_condition() {
    this->exit_condition = true;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    command.process();
    if (blocking) {
        this->finish(sub_device_ids);
    }
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer>& buffer, void* dst, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    this->enqueue_read_buffer(*buffer, dst, blocking, sub_device_ids);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion
// region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_read_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Read Buffer cannot be used with tracing");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());

    uint32_t padded_page_size = buffer.aligned_page_size();
    uint32_t pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    if (sub_device_ids.empty()) {
        sub_device_ids = tt::stl::Span<const SubDeviceId>(this->device->get_sub_device_ids());
    }

    if (is_sharded(buffer.buffer_layout())) {
        const bool width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
        const auto& buffer_page_mapping = width_split ? buffer.get_buffer_page_mapping() : nullptr;

        // Note that the src_page_index is the device page idx, not the host page idx
        // Since we read core by core we are reading the device pages sequentially
        const auto& cores = width_split ? buffer_page_mapping->all_cores_
                                        : corerange_to_cores(
                                              buffer.shard_spec().grid(),
                                              buffer.num_cores(),
                                              buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            uint32_t num_pages_to_read;
            if (width_split) {
                num_pages_to_read =
                    buffer_page_mapping->core_shard_shape_[core_id][0] * buffer.shard_spec().shape_in_pages()[1];
            } else {
                num_pages_to_read = std::min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages_to_read;
            }
            uint32_t bank_base_address = buffer.address();
            if (buffer.is_dram()) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            if (num_pages_to_read > 0) {
                if (width_split) {
                    uint32_t host_page = buffer_page_mapping->core_host_page_indices_[core_id][0];
                    src_page_index = buffer_page_mapping->host_page_to_dev_page_mapping_[host_page];
                    unpadded_dst_offset = host_page * buffer.page_size();
                } else {
                    unpadded_dst_offset = src_page_index * buffer.page_size();
                }

                auto command = EnqueueReadShardedBufferCommand(
                    this->id,
                    this->device,
                    this->noc_index,
                    buffer,
                    dst,
                    this->manager,
                    this->expected_num_workers_completed,
                    sub_device_ids,
                    cores[core_id],
                    bank_base_address,
                    src_page_index,
                    num_pages_to_read);

                this->issued_completion_q_reads.push(std::make_shared<detail::CompletionReaderVariant>(
                    std::in_place_type<detail::ReadBufferDescriptor>,
                    buffer.buffer_layout(),
                    buffer.page_size(),
                    padded_page_size,
                    dst,
                    unpadded_dst_offset,
                    num_pages_to_read,
                    src_page_index,
                    buffer_page_mapping));

                src_page_index += num_pages_to_read;
                this->enqueue_command(command, false, sub_device_ids);
                this->increment_num_entries_in_completion_q();
            }
        }
        if (blocking) {
            this->finish(sub_device_ids);
        }
    } else {
        if (pages_to_read > 0) {
            // this is a streaming command so we don't need to break down to multiple
            auto command = EnqueueReadInterleavedBufferCommand(
                this->id,
                this->device,
                this->noc_index,
                buffer,
                dst,
                this->manager,
                this->expected_num_workers_completed,
                sub_device_ids,
                src_page_index,
                pages_to_read);

            this->issued_completion_q_reads.push(std::make_shared<detail::CompletionReaderVariant>(
                std::in_place_type<detail::ReadBufferDescriptor>,
                buffer.buffer_layout(),
                buffer.page_size(),
                padded_page_size,
                dst,
                unpadded_dst_offset,
                pages_to_read,
                src_page_index));
            this->enqueue_command(command, blocking, sub_device_ids);
            this->increment_num_entries_in_completion_q();
        } else {
            if (blocking) {
                this->finish(sub_device_ids);
            }
        }
    }
}

void HWCommandQueue::enqueue_write_buffer(
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, HostDataType src, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    auto data = std::visit([&](auto&& data) -> const void* {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, const void*>) {
            return data;
        } else {
            return data->data();
        }
    }, src);
    auto& b = std::visit([&](auto&& b) -> Buffer& {
        using type_buf = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<type_buf, std::shared_ptr<Buffer>>) {
            return *b;
        } else {
            return b.get();
        }
    }, buffer);
    this->enqueue_write_buffer(b, data, blocking, sub_device_ids);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Write Buffer cannot be used with tracing");

    uint32_t padded_page_size = buffer.aligned_page_size();

    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    uint32_t max_data_sizeB =
        max_prefetch_command_size - (hal.get_alignment(HalMemType::HOST) * 2);  // * 2 to account for issue

    uint32_t dst_page_index = 0;

    if (sub_device_ids.empty()) {
        sub_device_ids = tt::stl::Span<const SubDeviceId>(this->device->get_sub_device_ids());
    }

    if (is_sharded(buffer.buffer_layout())) {
        const bool width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
        const auto& buffer_page_mapping = width_split ? buffer.get_buffer_page_mapping() : nullptr;

        const auto& cores = width_split ? buffer_page_mapping->all_cores_
                                        : corerange_to_cores(
                                              buffer.shard_spec().grid(),
                                              buffer.num_cores(),
                                              buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
        TT_FATAL(
            max_data_sizeB >= padded_page_size,
            "Writing padded page size > {} is currently unsupported for sharded tensors.",
            max_data_sizeB);
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();

        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            // Skip writing the padded pages along the bottom
            // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
            // Alternative write each page row into separate commands, or have a strided linear write
            uint32_t num_pages;
            if (width_split) {
                num_pages =
                    buffer_page_mapping->core_shard_shape_[core_id][0] * buffer.shard_spec().shape_in_pages()[1];
                if (num_pages == 0) {
                    continue;
                }
                dst_page_index =
                    buffer_page_mapping->host_page_to_dev_page_mapping_[buffer_page_mapping->core_host_page_indices_[core_id][0]];
            } else {
                num_pages = std::min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages;
            }
            uint32_t curr_page_idx_in_shard = 0;
            uint32_t bank_base_address = buffer.address();
            if (buffer.is_dram()) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            while (num_pages != 0) {
                // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
                uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
                bool issue_wait = dst_page_index == 0;  // only stall for the first write of the buffer
                if (issue_wait) {
                    // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                    data_offset_bytes *= 2;
                }
                uint32_t space_available_bytes = std::min(
                    command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
                int32_t num_pages_available =
                    (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(padded_page_size);

                uint32_t pages_to_write = std::min(num_pages, (uint32_t)num_pages_available);
                if (pages_to_write > 0) {
                    uint32_t address = bank_base_address + curr_page_idx_in_shard * padded_page_size;

                    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

                    auto command = EnqueueWriteShardedBufferCommand(
                        this->id,
                        this->device,
                        this->noc_index,
                        buffer,
                        src,
                        this->manager,
                        issue_wait,
                        this->expected_num_workers_completed,
                        sub_device_ids,
                        address,
                        buffer_page_mapping,
                        cores[core_id],
                        padded_page_size,
                        dst_page_index,
                        pages_to_write);

                    this->enqueue_command(command, false, sub_device_ids);
                    curr_page_idx_in_shard += pages_to_write;
                    num_pages -= pages_to_write;
                    dst_page_index += pages_to_write;
                } else {
                    this->manager.wrap_issue_queue_wr_ptr(this->id);
                }
            }
        }
    } else {
        uint32_t total_pages_to_write = buffer.num_pages();
        bool write_partial_pages = padded_page_size > max_data_sizeB;
        uint32_t page_size_to_write = padded_page_size;
        uint32_t padded_buffer_size = buffer.num_pages() * padded_page_size;
        if (write_partial_pages) {
            TT_FATAL(buffer.num_pages() == 1, "TODO: add support for multi-paged buffer with page size > 64KB");
            uint32_t partial_size = dispatch_constants::BASE_PARTIAL_PAGE_SIZE;
            uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
            while (padded_buffer_size % partial_size != 0) {
                partial_size += pcie_alignment;
            }
            page_size_to_write = partial_size;
            total_pages_to_write = padded_buffer_size / page_size_to_write;
        }

        const uint32_t num_banks = this->device->num_banks(buffer.buffer_type());
        uint32_t num_pages_round_robined = buffer.num_pages() / num_banks;
        uint32_t num_banks_with_residual_pages = buffer.num_pages() % num_banks;
        uint32_t num_partial_pages_per_page = padded_page_size / page_size_to_write;
        uint32_t num_partials_round_robined = num_partial_pages_per_page * num_pages_round_robined;

        uint32_t max_num_pages_to_write = write_partial_pages
                                              ? (num_pages_round_robined > 0 ? (num_banks * num_partials_round_robined)
                                                                             : num_banks_with_residual_pages)
                                              : total_pages_to_write;

        uint32_t bank_base_address = buffer.address();

        uint32_t num_full_pages_written = 0;
        while (total_pages_to_write > 0) {
            uint32_t data_offsetB = hal.get_alignment(HalMemType::HOST);  // data appended after CQ_PREFETCH_CMD_RELAY_INLINE
                                                                    // + CQ_DISPATCH_CMD_WRITE_PAGED
            bool issue_wait =
                (dst_page_index == 0 and
                 bank_base_address == buffer.address());  // only stall for the first write of the buffer
            if (issue_wait) {
                data_offsetB *= 2;  // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            }

            uint32_t space_availableB = std::min(
                command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
            int32_t num_pages_available =
                (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(page_size_to_write);

            if (num_pages_available <= 0) {
                this->manager.wrap_issue_queue_wr_ptr(this->id);
                continue;
            }

            uint32_t num_pages_to_write =
                std::min(std::min((uint32_t)num_pages_available, max_num_pages_to_write), total_pages_to_write);

            // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
            // To handle larger page offsets move bank base address up and update page offset to be relative to the new
            // bank address
            if (dst_page_index > 0xFFFF or (num_pages_to_write == max_num_pages_to_write and write_partial_pages)) {
                uint32_t num_banks_to_use = write_partial_pages ? max_num_pages_to_write : num_banks;
                uint32_t residual = dst_page_index % num_banks_to_use;
                uint32_t num_pages_written_per_bank = dst_page_index / num_banks_to_use;
                bank_base_address += num_pages_written_per_bank * page_size_to_write;
                dst_page_index = residual;
            }

            tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", this->id);

            auto command = EnqueueWriteInterleavedBufferCommand(
                this->id,
                this->device,
                this->noc_index,
                buffer,
                src,
                this->manager,
                issue_wait,
                this->expected_num_workers_completed,
                sub_device_ids,
                bank_base_address,
                page_size_to_write,
                dst_page_index,
                num_pages_to_write);
            this->enqueue_command(
                command, false, sub_device_ids);  // don't block until the entire src data is enqueued in the issue queue

            total_pages_to_write -= num_pages_to_write;
            dst_page_index += num_pages_to_write;
        }
    }

    if (blocking) {
        this->finish(sub_device_ids);
    }
}

void HWCommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
    std::vector<SubDeviceId> sub_device_ids = {program.determine_sub_device_ids(device)};
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    // Finalize Program: Compute relative offsets for data structures (semaphores, kernel binaries, etc) in L1
    if (not program.is_finalized()) {
        program.finalize(device);
    }
    if (program.get_program_binary_status(device->id()) == ProgramBinaryStatus::NotSent) {
        // Write program binaries to device if it hasn't previously been cached
        program.allocate_kernel_bin_buf_on_device(device);
        if (program.get_program_transfer_info().binary_data.size()) {
            this->enqueue_write_buffer(
                *program.get_kernels_buffer(device), program.get_program_transfer_info().binary_data.data(), false, sub_device_ids);
        }
        program.set_program_binary_status(device->id(), ProgramBinaryStatus::InFlight);
    }
    // Lower the program to device: Generate dispatch commands.
    // Values in these commands will get updated based on kernel config ring
    // buffer state at runtime.
    program.generate_dispatch_commands(device);
    program.set_last_used_command_queue_for_testing(this);

#ifdef DEBUG
    if (tt::llrt::RunTimeOptions::get_instance().get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (const auto buffer = program.get_kernels_buffer(device)) {
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            this->enqueue_read_buffer(*buffer, read_data.data(), true, sub_device_ids);
            TT_FATAL(
                program.get_program_transfer_info().binary_data == read_data,
                "Binary for program to be executed is corrupted. Another program likely corrupted this binary");
        }
    }
#endif
    auto sub_device_id = sub_device_ids[0];
    auto sub_device_index = sub_device_id.to_index();

    // Snapshot of expected workers from previous programs, used for dispatch_wait cmd generation.
    uint32_t expected_workers_completed = this->manager.get_bypass_mode() ? this->trace_ctx->descriptors[sub_device_id].num_completion_worker_cores
                                                                          : this->expected_num_workers_completed[sub_device_index];
    if (this->manager.get_bypass_mode()) {
        if (program.runs_on_noc_multicast_only_cores()) {
            this->trace_ctx->descriptors[sub_device_id].num_traced_programs_needing_go_signal_multicast++;
            this->trace_ctx->descriptors[sub_device_id].num_completion_worker_cores += device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            this->trace_ctx->descriptors[sub_device_id].num_traced_programs_needing_go_signal_unicast++;
            this->trace_ctx->descriptors[sub_device_id].num_completion_worker_cores += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
    } else {
        if (program.runs_on_noc_multicast_only_cores()) {
            this->expected_num_workers_completed[sub_device_index] += device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            this->expected_num_workers_completed[sub_device_index] += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
    }

    auto &worker_launch_message_buffer_state = this->device->get_worker_launch_message_buffer_state(sub_device_id);
    auto command = EnqueueProgramCommand(
        this->id,
        this->device,
        this->noc_index,
        program,
        this->virtual_enqueue_program_dispatch_core,
        this->manager,
        this->get_config_buffer_mgr(sub_device_index),
        expected_workers_completed,
        // The assembled program command will encode the location of the launch messages in the ring buffer
        worker_launch_message_buffer_state.get_mcast_wptr(),
        worker_launch_message_buffer_state.get_unicast_wptr(),
        sub_device_id);
    // Update wptrs for tensix and eth launch message in the device class
    if (program.runs_on_noc_multicast_only_cores()) {
        worker_launch_message_buffer_state.inc_mcast_wptr(1);
    }
    if (program.runs_on_noc_unicast_only_cores()) {
        worker_launch_message_buffer_state.inc_unicast_wptr(1);
    }
    this->enqueue_command(command, blocking, sub_device_ids);

#ifdef DEBUG
    if (tt::llrt::RunTimeOptions::get_instance().get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (const auto buffer = program.get_kernels_buffer(device)) {
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            this->enqueue_read_buffer(*buffer, read_data.data(), true, sub_device_ids);
            TT_FATAL(
                program.get_program_transfer_info().binary_data == read_data,
                "Binary for program that executed is corrupted. This program likely corrupted its own binary.");
        }
    }
#endif

    log_trace(
        tt::LogMetal,
        "Created EnqueueProgramCommand (active_cores: {} bypass_mode: {} expected_workers_completed: {})",
        program.get_program_transfer_info().num_active_cores,
        this->manager.get_bypass_mode(),
        expected_workers_completed);
}

void HWCommandQueue::enqueue_record_event(const std::shared_ptr<Event>& event, bool clear_count, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Record Event cannot be used with tracing");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true;  // what does this mean???

    if (sub_device_ids.empty()) {
        sub_device_ids = tt::stl::Span<const SubDeviceId>(this->device->get_sub_device_ids());
    }

    auto command = EnqueueRecordEventCommand(
        this->id,
        this->device,
        this->noc_index,
        this->manager,
        event->event_id,
        this->expected_num_workers_completed,
        sub_device_ids,
        clear_count,
        true);
    this->enqueue_command(command, false, sub_device_ids);

    if (clear_count) {
        for (const auto& id : sub_device_ids) {
            this->expected_num_workers_completed[id.to_index()] = 0;
        }
    }
    this->issued_completion_q_reads.push(
        std::make_shared<detail::CompletionReaderVariant>(std::in_place_type<detail::ReadEventDescriptor>, event->event_id));
    this->increment_num_entries_in_completion_q();
}

void HWCommandQueue::enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    auto command = EnqueueWaitForEventCommand(this->id, this->device, this->manager, *sync_event, clear_count);
    this->enqueue_command(command, false, {});

    if (clear_count) {
        this->manager.reset_event_id(this->id);
    }
}

void HWCommandQueue::enqueue_trace(const uint32_t trace_id, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_trace");

    auto trace_inst = this->device->get_trace(trace_id);
    auto command = EnqueueTraceCommand(
        this->id, this->device, this->manager, trace_inst->desc, *trace_inst->buffer, this->expected_num_workers_completed, this->noc_index, this->virtual_enqueue_program_dispatch_core);

    this->enqueue_command(command, false, {});

    for (const auto& [id, desc]: trace_inst->desc->descriptors) {
        auto index = id.to_index();
        // Increment the expected worker cores counter due to trace programs completion
        this->expected_num_workers_completed[index] += desc.num_completion_worker_cores;
        // After trace runs, the rdptr on each worker will be incremented by the number of programs in the trace
        // Update the wptr on host to match state. If the trace doesn't execute on a
        // class of worker (unicast or multicast), it doesn't reset or modify the
        // state for those workers.
        auto &worker_launch_message_buffer_state = this->device->get_worker_launch_message_buffer_state(id);
        if (desc.num_traced_programs_needing_go_signal_multicast) {
            worker_launch_message_buffer_state.set_mcast_wptr(desc.num_traced_programs_needing_go_signal_multicast);
        }
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            worker_launch_message_buffer_state.set_unicast_wptr(desc.num_traced_programs_needing_go_signal_unicast);
        }
        // The config buffer manager is unaware of what memory is used inside the trace, so mark all memory as used so that
        // it will force a stall and avoid stomping on in-use state.
        // TODO(jbauman): Reuse old state from the trace.
        this->config_buffer_mgr[index].mark_completely_full(this->expected_num_workers_completed[index]);
    }
    if (blocking) {
        this->finish(trace_inst->desc->sub_device_ids);
    }
}

void HWCommandQueue::copy_into_user_space(
    const detail::ReadBufferDescriptor& read_buffer_descriptor, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, buffer_page_mapping, dst, dst_offset, num_pages_read, cur_dev_page_id] =
        read_buffer_descriptor;
    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;
    uint32_t dev_page_id = cur_dev_page_id;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    std::optional<uint32_t> host_page_id = std::nullopt;
    uint32_t offset_in_completion_q_data = sizeof(CQDispatchCmd);

    uint32_t pad_size_bytes = padded_page_size - page_size;

    while (remaining_bytes_to_read != 0) {
        uint32_t completion_queue_write_ptr_and_toggle = this->manager.completion_queue_wait_front(this->id, this->exit_condition);

        if (this->exit_condition) {
            break;
        }

        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
        uint32_t completion_q_read_toggle = this->manager.get_completion_queue_read_toggle(this->id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue =
                this->manager.get_completion_queue_limit(this->id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(remaining_bytes_to_read, bytes_avail_in_completion_queue);
        uint32_t num_pages_xfered = div_up(bytes_xfered, dispatch_constants::TRANSFER_PAGE_SIZE);

        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_page_mapping == nullptr) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            if (page_size == padded_page_size) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data;
                tt::Cluster::instance().read_sysmem(
                    contiguous_dst,
                    data_bytes_xfered,
                    completion_q_read_ptr + offset_in_completion_q_data,
                    mmio_device_id,
                    channel);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset_bytes = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint32_t dst_offset_bytes = 0;

                while (src_offset_bytes < bytes_xfered) {
                    uint32_t src_offset_increment = padded_page_size;
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = num_bytes_to_copy;
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes;
                                // Only pad data left in queue
                            } else {
                                offset_in_completion_q_data = pad_size_bytes - rem_bytes_in_cq;
                            }
                        }
                    } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    tt::Cluster::instance().read_sysmem(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        num_bytes_to_copy,
                        completion_q_read_ptr + src_offset_bytes,
                        mmio_device_id,
                        channel);

                    src_offset_bytes += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else {
            uint32_t src_offset_bytes = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint32_t dst_offset_bytes = contig_dst_offset;
            uint32_t num_bytes_to_copy = 0;

            while (src_offset_bytes < bytes_xfered) {
                uint32_t src_offset_increment = padded_page_size;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = num_bytes_to_copy;
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        dev_page_id++;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes;
                            offset_in_completion_q_data = 0;
                            // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq);
                        }
                    }
                    if (!host_page_id.has_value()) {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    host_page_id = buffer_page_mapping->dev_page_to_host_page_mapping_[dev_page_id];
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        dev_page_id++;
                    }
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = *host_page_id * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    host_page_id = buffer_page_mapping->dev_page_to_host_page_mapping_[dev_page_id];
                    dev_page_id++;
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = *host_page_id * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                }

                tt::Cluster::instance().read_sysmem(
                    (char*)(uint64_t(dst) + dst_offset_bytes),
                    num_bytes_to_copy,
                    completion_q_read_ptr + src_offset_bytes,
                    mmio_device_id,
                    channel);

                src_offset_bytes += src_offset_increment;
            }
            dst_offset_bytes += num_bytes_to_copy;
            contig_dst_offset = dst_offset_bytes;
        }
        this->manager.completion_queue_pop_front(num_pages_xfered, this->id);
    }
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        {
            std::unique_lock<std::mutex> lock(this->reader_thread_cv_mutex);
            this->reader_thread_cv.wait(lock, [this] {
                return this->num_entries_in_completion_q > this->num_completed_completion_q_reads or
                       this->exit_condition;
            });
        }
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            ZoneScopedN("CompletionQueueReader");
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {
                ZoneScopedN("CompletionQueuePopulated");
                auto read_descriptor = *(this->issued_completion_q_reads.pop());
                {
                    ZoneScopedN("CompletionQueueWait");
                    this->manager.completion_queue_wait_front(
                        this->id, this->exit_condition);  // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN
                }
                if (this->exit_condition) {  // Early exit
                    return;
                }

                std::visit(
                    [&](auto&& read_descriptor) {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            ZoneScopedN("CompletionQueueReadData");
                            this->copy_into_user_space(read_descriptor, mmio_device_id, channel);
                        } else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            ZoneScopedN("CompletionQueueReadEvent");
                            uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
                            thread_local static std::vector<uint32_t> dispatch_cmd_and_event(
                                (sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(),
                                sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
                                read_ptr,
                                mmio_device_id,
                                channel);
                            uint32_t event_completed =
                                dispatch_cmd_and_event[sizeof(CQDispatchCmd) / sizeof(uint32_t)];

                            TT_ASSERT(
                                event_completed == read_descriptor.event_id,
                                "Event Order Issue: expected to read back completion signal for event {} but got {}!",
                                read_descriptor.event_id,
                                event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, read_descriptor.get_global_event_id());
                            log_trace(
                                LogAlways,
                                "Completion queue popped event {} (global: {})",
                                event_completed,
                                read_descriptor.get_global_event_id());
                        }
                    },
                    read_descriptor);
            }
            this->num_completed_completion_q_reads += num_events_to_read;
            {
                std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex);
                this->reads_processed_cv.notify_one();
            }
        } else if (this->exit_condition) {
            return;
        }
    }
}

void HWCommandQueue::finish(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event, false, sub_device_ids);
    if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->dprint_server_hang = true;
                this->set_exit_condition();
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->illegal_noc_txn_hang = true;
                this->set_exit_condition();
                return;
            }
        }
    } else {
        std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex);
        this->reads_processed_cv.wait(
            lock, [this] { return this->num_entries_in_completion_q == this->num_completed_completion_q_reads; });
    }
}

volatile bool HWCommandQueue::is_dprint_server_hung() { return dprint_server_hang; }

volatile bool HWCommandQueue::is_noc_hung() { return illegal_noc_txn_hang; }

void HWCommandQueue::record_begin(const uint32_t tid, std::shared_ptr<detail::TraceDescriptor> ctx) {
    uint32_t num_sub_devices = this->device->num_sub_devices();
    // Issue event as a barrier and a counter reset
    uint32_t cmd_sequence_sizeB = hal.get_alignment(HalMemType::HOST);
    if (this->device->distributed_dispatcher()) {
        // wait on dispatch_s before issuing counter reset
        cmd_sequence_sizeB += hal.get_alignment(HalMemType::HOST);
    }
    cmd_sequence_sizeB *= num_sub_devices;
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint32_t dispatch_message_base_addr = dispatch_constants::get(
        dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);

    // Currently Trace will track all sub_devices
    // Potentially support tracking only used sub_devices in the future
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        uint32_t dispatch_message_addr = dispatch_message_base_addr + dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(i);
        if (this->device->distributed_dispatcher()) {
            // wait on dispatch_s before issuing counter reset
            command_sequence.add_dispatch_wait(false, dispatch_message_addr, this->expected_num_workers_completed[i], true, false, true, 1);
        }
        // dispatch_d waits for latest non-zero counter from dispatch_s and then clears its local counter
        command_sequence.add_dispatch_wait(false, dispatch_message_addr, this->expected_num_workers_completed[i], true);
    }

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->id);
    this->manager.fetch_queue_reserve_back(this->id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->id);
    std::fill(this->expected_num_workers_completed.begin(), this->expected_num_workers_completed.begin() + num_sub_devices, 0);
    // Record commands using bypass mode
    this->tid = tid;
    this->trace_ctx = std::move(ctx);
    // Record original value of launch msg wptr
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        auto &worker_launch_message_buffer_state = this->device->get_worker_launch_message_buffer_state(SubDeviceId{i});
        this->multicast_cores_launch_message_wptr_reset[i] = worker_launch_message_buffer_state.get_mcast_wptr();
        this->unicast_cores_launch_message_wptr_reset[i] = worker_launch_message_buffer_state.get_unicast_wptr();
        // Set launch msg wptr to 0. Every time trace runs on device, it will ensure that the workers
        // reset their rptr to be in sync with device.
        worker_launch_message_buffer_state.reset();
    }
    this->manager.set_bypass_mode(true, true);  // start
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        // Sync values in the trace need to match up with the counter starting at 0 again.
        this->config_buffer_mgr[i].mark_completely_full(this->expected_num_workers_completed[i]);
    }
}

void HWCommandQueue::record_end() {
    auto &trace_data = this->trace_ctx->data;
    trace_data = std::move(this->manager.get_bypass_data());
    // Add command to terminate the trace buffer
    DeviceCommand command_sequence(hal.get_alignment(HalMemType::HOST));
    command_sequence.add_prefetch_exec_buf_end();
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        trace_data.push_back(((uint32_t*)command_sequence.data())[i]);
    }
    // Currently Trace will track all sub_devices
    uint32_t num_sub_devices = this->device->num_sub_devices();
    // Reset the launch msg wptrs to their original value, so device can run programs after a trace
    // was captured. This is needed since trace capture modifies the wptr state on host, even though device
    // doesn't run any programs.
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        auto &worker_launch_message_buffer_state = this->device->get_worker_launch_message_buffer_state(SubDeviceId{i});
        worker_launch_message_buffer_state.set_mcast_wptr(this->multicast_cores_launch_message_wptr_reset[i]);
        worker_launch_message_buffer_state.set_unicast_wptr(this->unicast_cores_launch_message_wptr_reset[i]);
    }
    // Copy the desc keys into a separate vector. When enqueuing traces, we sometimes need to pass sub-device ids separately
    this->trace_ctx->sub_device_ids.reserve(this->trace_ctx->descriptors.size());
    for (const auto& [id, _]: this->trace_ctx->descriptors) {
        auto index = id.to_index();
        this->trace_ctx->sub_device_ids.push_back(id);
        // config_buffer_mgr reflects the state inside the trace, not on the current device, so reset it.
        // TODO(jbauman): Use a temporary WorkingBufferSetMgr when recording a trace.
        this->get_config_buffer_mgr(index).mark_completely_full(this->expected_num_workers_completed[index]);
    }
    this->tid = std::nullopt;
    this->trace_ctx = nullptr;
    this->manager.set_bypass_mode(false, true);  // stop
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager.get_bypass_mode(), "Terminate cannot be used with tracing");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id);
    auto command = EnqueueTerminateCommand(this->id, this->device, this->manager);
    this->enqueue_command(command, false, {});
}

WorkerConfigBufferMgr& HWCommandQueue::get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr[index]; }

void HWCommandQueue::reset_config_buffer_mgr(const uint32_t num_entries) {
    for (uint32_t i = 0; i < num_entries; ++i) {
        this->config_buffer_mgr[i] = WorkerConfigBufferMgr();
        for (uint32_t index = 0; index < tt::tt_metal::hal.get_programmable_core_type_count(); index++) {
            this->config_buffer_mgr[i].init_add_buffer(
                tt::tt_metal::hal.get_dev_addr(
                    tt::tt_metal::hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG),
                tt::tt_metal::hal.get_dev_size(
                    tt::tt_metal::hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG));
        }
        // Subtract 1 from the number of entries, so the watcher can read information (e.g. fired asserts) from the
        // previous launch message.
        this->config_buffer_mgr[i].init_add_buffer(0, launch_msg_buffer_num_entries - 1);

        // There's no ring buffer for active ethernet binaries, so keep track of them separately.
        this->config_buffer_mgr[i].init_add_buffer(0, 1);
    }
}

void EnqueueAddBufferToProgramImpl(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    Program& program) {
    std::visit(
        [&program](auto&& b) {
            using buffer_type = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
                program.add_buffer(b);
            }
        },
        buffer);
}

void EnqueueAddBufferToProgram(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    Program& program,
    bool blocking) {
    EnqueueAddBufferToProgramImpl(std::move(buffer), program);
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit(
            [&resolved_runtime_args](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, Buffer*>) {
                    resolved_runtime_args.push_back(a->address());
                } else {
                    resolved_runtime_args.push_back(a);
                }
            },
            arg);
    }
    runtime_args_md.kernel->set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueSetRuntimeArgs(
    CommandQueue& cq,
    const std::shared_ptr<Kernel>& kernel,
    const CoreCoord& core_coord,
    std::shared_ptr<RuntimeArgs> runtime_args_ptr,
    bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata{
        .core_coord = core_coord,
        .runtime_args_ptr = std::move(runtime_args_ptr),
        .kernel = kernel,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer->address();
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::GET_BUF_ADDR, .blocking = blocking, .shadow_buffer = buffer, .dst = dst_buf_addr});
}

inline namespace v0 {

void EnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    std::vector<uint32_t>& src,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, std::move(buffer), src.data(), blocking, sub_device_ids);
}

void EnqueueReadBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER, .blocking = blocking, .buffer = buffer, .dst = dst, .sub_device_ids = sub_device_ids});
}

void EnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER, .blocking = blocking, .buffer = buffer, .src = std::move(src), .sub_device_ids = sub_device_ids});
}

void EnqueueProgram(
    CommandQueue& cq, Program& program, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::ENQUEUE_PROGRAM, .blocking = blocking, .program = &program});
}

void EnqueueRecordEvent(CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event,
        .sub_device_ids = sub_device_ids
    });
}

void EnqueueWaitForEvent(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EventSynchronize(const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    log_trace(
        tt::LogMetal,
        "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})",
        event->device->id(),
        event->cq_id,
        event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_FATAL(
                false,
                "Command Queue could not complete EventSynchronize. See {} for details.",
                tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool EventQuery(const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        event->device->id(),
        event->cq_id,
        event->event_id);
    return event_completed;
}

void Finish(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{.type = EnqueueCommandType::FINISH, .blocking = true, .sub_device_ids = sub_device_ids});
    TT_ASSERT(
        !(cq.device()->hw_command_queue(cq.id()).is_dprint_server_hung()),
        "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(
        !(cq.device()->hw_command_queue(cq.id()).is_noc_hung()),
        "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
        tt::watcher_get_log_file_name());
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(cq.device()->get_trace(trace_id) != nullptr, "Trace instance {} must exist on device", trace_id);
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::ENQUEUE_TRACE, .blocking = blocking, .trace_id = trace_id});
}

}  // namespace v0

void EnqueueReadBufferImpl(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    std::visit(
        [&](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking, sub_device_ids);
            }
        },
        buffer);
}

void EnqueueWriteBufferImpl(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    cq.hw_command_queue().enqueue_write_buffer(std::move(buffer), std::move(src), blocking, sub_device_ids);
}

void EnqueueProgramImpl(
    CommandQueue& cq, Program& program, bool blocking) {
    ZoneScoped;

    Device* device = cq.device();
    detail::CompileProgram(device, program);
    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);
    cq.hw_command_queue().enqueue_program(program, blocking);
    // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem
    // leaks on device.
    program.release_buffers();

}

void EnqueueRecordEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    cq.hw_command_queue().enqueue_record_event(event, false, sub_device_ids);
}

void EnqueueWaitForEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    event->wait_until_ready();  // Block until event populated. Worker thread.
    log_trace(
        tt::LogMetal,
        "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(),
        event->cq_id,
        event->event_id,
        cq.device()->id(),
        cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}

void FinishImpl(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) { cq.hw_command_queue().finish(sub_device_ids); }

void EnqueueTraceImpl(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    cq.hw_command_queue().enqueue_trace(trace_id, blocking);
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device), cq_id(id), mode(mode), worker_state(CommandQueueState::IDLE) {
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace& trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (not this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() { return this->device()->hw_command_queue(this->cq_id); }

void CommandQueue::dump() {
    int cid = 0;
    log_info(LogMetalTrace, "Dumping {}, mode={}", this->name(), this->get_mode());
    for (const auto& cmd : this->worker_queue) {
        log_info(LogMetalTrace, "[{}]: {}", cid, cmd.type);
        cid++;
    }
}

std::string CommandQueue::name() {
    if (this->mode == CommandQueueMode::TRACE) {
        return "TraceQueue";
    }
    return "CQ" + std::to_string(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "{} WFI start", this->name());
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "{} WFI complete", this->name());
}

void CommandQueue::set_mode(const CommandQueueMode& mode) {
    TT_ASSERT(
        not this->trace_mode(),
        "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "{} started worker thread", this->name());
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "{} stopped worker thread", this->name());
}

void CommandQueue::run_worker() {
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(CommandInterface&& command) {
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
    if (this->async_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id) {
            // In async mode when parent pushes cmd, feed worker through queue.
            bool blocking = command.blocking.has_value() and *command.blocking;
            this->worker_queue.push(std::move(command));
            if (blocking) {
                TT_ASSERT(not this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        } else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(
                std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id,
                "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else if (this->trace_mode()) {
        // In trace mode push to the trace queue
        this->worker_queue.push(std::move(command));
    } else if (this->passthrough_mode()) {
        this->run_command_impl(command);
    } else {
        TT_THROW("Unsupported CommandQueue mode!");
    }
}

void CommandQueue::run_command_impl(const CommandInterface& command) {
    log_trace(LogDispatch, "{} running {}", this->name(), command.type);
    switch (command.type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueReadBufferImpl(*this, command.buffer.value(), command.dst.value(), command.blocking.value(), command.sub_device_ids);
            break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
            TT_ASSERT(command.src.has_value(), "Must provide a src!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueWriteBufferImpl(*this, command.buffer.value(), command.src.value(), command.blocking.value(), command.sub_device_ids);
            break;
        case EnqueueCommandType::GET_BUF_ADDR:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
            TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
            EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
            break;
        case EnqueueCommandType::SET_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program != nullptr, "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), *command.program);
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program != nullptr, "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, *command.program, command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_TRACE:
            EnqueueTraceImpl(*this, command.trace_id.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueRecordEventImpl(*this, command.event.value(), command.sub_device_ids);
            break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueWaitForEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::FINISH: FinishImpl(*this, command.sub_device_ids); break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default: TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "{} running {} complete", this->name(), command.type);
}

v1::CommandQueueHandle v1::GetCommandQueue(DeviceHandle device, std::uint8_t cq_id) {
    return v1::CommandQueueHandle{device, cq_id};
}

v1::CommandQueueHandle v1::GetDefaultCommandQueue(DeviceHandle device) { return GetCommandQueue(device, 0); }

void v1::EnqueueReadBuffer(CommandQueueHandle cq, const BufferHandle& buffer, std::byte *dst, bool blocking) {
    v0::EnqueueReadBuffer(GetDevice(cq)->command_queue(GetId(cq)), *buffer, dst, blocking);
}

void v1::EnqueueWriteBuffer(CommandQueueHandle cq, const BufferHandle& buffer, const std::byte *src, bool blocking) {
    v0::EnqueueWriteBuffer(GetDevice(cq)->command_queue(GetId(cq)), *buffer, src, blocking);
}

void v1::EnqueueProgram(CommandQueueHandle cq, ProgramHandle &program, bool blocking) {
    v0::EnqueueProgram(GetDevice(cq)->command_queue(GetId(cq)), program, blocking);
}

void v1::Finish(CommandQueueHandle cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    v0::Finish(GetDevice(cq)->command_queue(GetId(cq)));
}

void v1::SetLazyCommandQueueMode(bool lazy) {
    detail::SetLazyCommandQueueMode(lazy);
}

v1::DeviceHandle v1::GetDevice(CommandQueueHandle cq) {
    return cq.device;
}

std::uint8_t v1::GetId(CommandQueueHandle cq) {
    return cq.id;
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type) {
    switch (type) {
        case CommandQueue::CommandQueueMode::PASSTHROUGH: os << "PASSTHROUGH"; break;
        case CommandQueue::CommandQueueMode::ASYNC: os << "ASYNC"; break;
        case CommandQueue::CommandQueueMode::TRACE: os << "TRACE"; break;
        default: TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}
