// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is generated by compile_sparse_attention.py using triton AoT compiler

#pragma once
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace sparse_attention_v1 {

// launcher for: sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2
Status sparse_attention_fp16_sm80_bef12fb0(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_bef12fb0(params);
}

// load for: sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2
void load_sparse_attention_fp16_sm80_bef12fb0();
void load_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2() {
  load_sparse_attention_fp16_sm80_bef12fb0();
}

// unload for: sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2
void unload_sparse_attention_fp16_sm80_bef12fb0();
void unload_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2() {
  unload_sparse_attention_fp16_sm80_bef12fb0();
}

// launcher for: sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2
Status sparse_attention_fp16_sm80_d7f3a63f(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_d7f3a63f(params);
}

// load for: sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2
void load_sparse_attention_fp16_sm80_d7f3a63f();
void load_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2() {
  load_sparse_attention_fp16_sm80_d7f3a63f();
}

// unload for: sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2
void unload_sparse_attention_fp16_sm80_d7f3a63f();
void unload_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2() {
  unload_sparse_attention_fp16_sm80_d7f3a63f();
}

typedef Status (*kernel_func_t)(SparseAttentionParams& params);
kernel_func_t sparse_attention_fp16_sm80_kernels[] = {
    sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2,
    sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2,
};

int sparse_attention_fp16_sm80_get_num_algos(void) {
  return (int)sizeof(sparse_attention_fp16_sm80_kernels);
}

Status sparse_attention_fp16_sm80(SparseAttentionParams& params, int algo_id) {
  assert(algo_id < (int)sizeof(sparse_attention_fp16_sm80_kernels));
  return sparse_attention_fp16_sm80_kernels[algo_id](params);
}

void load_sparse_attention_fp16_sm80(void) {
  load_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2();
  load_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2();
}

void unload_sparse_attention_fp16_sm80(void) {
  unload_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2();
  unload_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2();
}

Status sparse_attention_fp16_sm80_default(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80(params, 0);
}

}  // namespace sparse_attention_v1
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
