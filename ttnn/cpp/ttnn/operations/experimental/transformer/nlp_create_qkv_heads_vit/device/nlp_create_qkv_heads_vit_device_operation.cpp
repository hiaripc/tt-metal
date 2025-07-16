// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit_device_operation.hpp"

#include "tt_metal/common/work_split.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::transformer {

// Hard-coded for Vit
void NlpCreateHeadsVitDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Error");

    TT_FATAL(input_shape[2] % tt::constants::TILE_HEIGHT == 0, "Error");
    TT_FATAL(
        (input_shape == tt::tt_metal::LegacyShape({input_shape[0], 1, input_shape[2], 2304})),
        "Unsupported input shape");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
}

std::vector<tt::tt_metal::LegacyShape> NlpCreateHeadsVitDeviceOperation::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    std::vector<tt::tt_metal::LegacyShape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    output_shape_vec = {
        (tt::tt_metal::LegacyShape){input_shape[0], 12, input_shape[2], 64},
        (tt::tt_metal::LegacyShape){input_shape[0], 12, input_shape[2], 64},
        (tt::tt_metal::LegacyShape){input_shape[0], 12, input_shape[2], 64}};
    return output_shape_vec;
}

std::vector<Tensor> NlpCreateHeadsVitDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpCreateHeadsVitDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return multi_core_nlp_create_qkv_heads_vit(input_tensor, output_tensors, compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::transformer
