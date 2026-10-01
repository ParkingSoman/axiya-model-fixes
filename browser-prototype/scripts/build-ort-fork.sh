#!/usr/bin/env bash
# Builds onnxruntime-web's WebGPU wasm (ort-wasm-simd-threaded.asyncify.{wasm,mjs})
# from the onnxruntime v1.30.0 tag, with or without the batched-int4 patch.
#
# The patch (reference/onnxruntime-webgpu.patch) gives every row of a short
# multi-token pass (2-16 tokens, a speculative verify pass) the same arithmetic
# as a one-token decode step, and makes those passes cheap:
#   - GroupQueryAttention decode (1-16 new tokens): FlashAttentionDecodeGQA,
#     one workgroup per (KV head, KV tile) for all query rows, fixed split-K
#     (K3). One-token steps use it too.
#   - q/k RMS norm folded into the rotary kernel for 1-16 tokens (K3).
#   - LayerNorm/SkipLayerNorm: one workgroup per row for every row count; the
#     one-row split path is off (K3). RMSNormalization too (PK): gemma12b's
#     global-layer k_norm is one row of 512 at S = 1.
#   - MatMulNBits, 1-16 rows (K4): one pipeline per path, DP4A (int8
#     activations) where one-token steps use DP4A, else float (float32 or
#     float16); each weight is read once per tile of rows. One-token steps
#     use it too. DP4A keeps the rows in a runtime loop. The float kernel (PK,
#     tile9) keeps up to 9 row accumulators in registers, each row updated by
#     its own inlined copy of one helper function; rows 10-16 take a second
#     tile. 16 accumulators spilled on Intel gen-9 (GLM-OCR decode step 227 ->
#     144 ms with 9). Bit identity across those row copies is verified on the
#     Apple M5 (gates e4b 528/528, 12B 92/92).
#   - Flash-attention prefill with 32-bit inputs steps 8 keys at a time
#     (tile9k8): fewer private registers per thread; the GLM-OCR image reader
#     ran 2.2x faster on Intel gen-9. 16-bit inputs (Gemma) keep their step. Other 2-16-row cases run the one-row
#     float program per row, never the prefill tile kernels.
#   - CausalConvWithState with state_window (K5): the WebGPU kernel keeps the
#     conv state after each of the last W positions (W <= 16), so a speculative
#     pass can roll back to an accepted prefix. The app fuses the decoder's
#     conv layers into this op at load (VITE_CONV_STATE, src/conv-state.js).
#   - LinearAttention on WebGPU (ovis, fork branch ovis-speed): state_window W
#     (<= 16) keeps the recurrent state after each of the last W tokens
#     (state tensors (W, B, H, dk, dv); the op reads slot W-1), so an
#     OvisOCR2 verify pass can roll back to an accepted prefix. Passes of more
#     than 16 tokens (the prompt step) take a chunked gated-delta prefill
#     (LinearAttentionChunkPrep + LinearAttentionChunkScan); passes of 1-16
#     tokens keep the token loop, so each row has the arithmetic of a
#     one-token step.
#   - Tiled fp16 attention (ovis): a session that sets the config entry
#     webgpu.flash_attention_tiled=1 runs the 16-bit, head-64 MultiHeadAttention
#     prefill as FlashAttentionTiled. Only the OvisOCR2 image reader sets it
#     (lock runtime.visionAttention 'tiled'); other sessions keep the default
#     programs.
#   - Fused MatMulNBitsMlp/Qkv one-token decode programs: kept by default
#     (FUSED_DECODE below).
#   - SplitMaskedSoftmax (P5, com.microsoft, WebGPU only): the gemma12b
#     split-cache attention's Concat + mask Add + Softmax + Cast(float16) + two
#     Slices in one pass (tools/gemma12b/fuse_split_softmax.py). Same softmax
#     arithmetic as the WebGPU Softmax program.
#   - 8-bit wide-tile MatMulNBits prefill (q8): the two vec4<u32> loads of one
#     8-bit block stay a struct (BData8) read with select; an
#     array<vec2<u32>, 4> indexed by the b_idx loop ran about 14x slower on the
#     M5 with the same output.
#   - Weight upload (PM, JSPI loader): an external initializer over 16 MB is
#     read from its Blob and uploaded in 16 MB pieces, and the upload waits for
#     the GPU queue after every 64 MB. Chrome never shrinks the shared-memory
#     pool behind staging buffers; before, it kept about 1 GB in the GPU
#     process after the Gemma 4 12B load, and the loader's JavaScript scratch
#     buffer kept the size of the largest initializer (1 GB).
#   - GPU-neutral kernels (GN): kernels see an empty adapter info (no GPU
#     brand, architecture or subgroup width), the device has no subgroup
#     features, and the limits kernels choose from are the WebGPU defaults,
#     so every GPU runs the same programs (docs/LOWMEM.md "GPU-neutral
#     kernels"). The patch also carries the E4B/E2B fp32 accumulation.
#
#   ORT_SRC=/path/to/ort-fork  PATCH=1  OUT=/path/to/out  scripts/build-ort-fork.sh
#
# JSPI build (gemma12b, P2a): besides the asyncify file, ORT can also build a
# JavaScript Promise Integration wasm (--enable_wasm_jspi, onnxruntime
# cmake/onnxruntime_webassembly.cmake). It needs the same source tree and
# patch, just a different link flag and its own cmake build folder, and
# produces ort-wasm-simd-threaded.jspi.{wasm,mjs} (the OUTPUT_NAME the cmake
# file picks for onnxruntime_ENABLE_WEBASSEMBLY_JSPI=ON) instead of the
# .asyncify pair. The worker gives gemma12b's decoder shards to ORT as Blobs
# in externalData (Phase 0.2); JSPI lets those awaits happen without the
# asyncify stack-switching overhead. Built by default, next to the asyncify
# build, unless JSPI=0.
#
# Env:
#   ORT_SRC  checkout folder (cloned here if missing; shallow, with shallow submodules)
#   PATCH    1 = apply reference/onnxruntime-webgpu.patch (default), 0 = unmodified tag
#   OUT      folder that receives a full onnxruntime-web dist: the npm 1.30.0 dist
#            files, with the asyncify files (and, unless JSPI=0, the jspi files)
#            replaced by this build
#   BUILD_DIR  cmake build folder for the asyncify build (default $ORT_SRC/build/wasm_webgpu)
#   JSPI     1 = also build ort-wasm-simd-threaded.jspi.{wasm,mjs} (default 1), 0 = skip it
#   BUILD_DIR_JSPI  cmake build folder for the JSPI build (default $ORT_SRC/build/wasm_webgpu_jspi)
#   PATCH_FILE patch to apply when PATCH=1 (default reference/onnxruntime-webgpu.patch)
#   FORCE_DP4A 1 = debug build: the DP4A MatMulNBits path also runs on Apple GPUs
#            (sets ORT_FORK_DEBUG_FORCE_DP4A in dp4a_matmul_nbits.cc), so a Mac
#            runs the int8-activation arithmetic of Intel/AMD/NVIDIA GPUs. Needs
#            PATCH=1. Test builds only; never ship one. Default 0.
#   FUSED_DECODE 1 = one-token steps may use the fused MatMulNBitsMlp/Qkv decode
#            programs (default). 0 = they take the same unfused path as 2-16-row
#            passes (sets ORT_FORK_FUSED_DECODE in dp4a_matmul_nbits.h): the
#            whole decoder is then bit-identical per row on Apple GPUs too, and
#            one-token steps cost +2% to +9% on the M5 (evidence/k4-matmulnbits).
#            Needs PATCH=1.
#
# The build flags are the ones the npm release uses for this file
# (tools/ci_build/github/azure-pipelines/templates/{web-ci,linux-wasm-ci}.yml,
# stage Build_wasm_Release, step "WebGPU experimental"). emsdk is the version
# the tag pins (4.0.23, cmake/external/emsdk).
#
# Needs: git, python3 (a venv with flatbuffers is made next to ORT_SRC), cmake >= 3.28, ninja, node.
# Disk: about 2 GB for the checkout, 2 GB for emsdk, 3-5 GB for the build folder.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
app="$(cd "$here/.." && pwd)"
TAG=v1.30.0
EMSDK_VERSION=4.0.23
ORT_SRC="${ORT_SRC:?set ORT_SRC to the onnxruntime checkout folder}"
PATCH="${PATCH:-1}"
FORCE_DP4A="${FORCE_DP4A:-0}"
FUSED_DECODE="${FUSED_DECODE:-1}"
OUT="${OUT:?set OUT to the output dist folder}"
BUILD_DIR="${BUILD_DIR:-$ORT_SRC/build/wasm_webgpu}"
JSPI="${JSPI:-1}"
BUILD_DIR_JSPI="${BUILD_DIR_JSPI:-$ORT_SRC/build/wasm_webgpu_jspi}"

if [ ! -d "$ORT_SRC/.git" ] && [ ! -f "$ORT_SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "$ORT_SRC"
fi
cd "$ORT_SRC"
test "$(git describe --tags --exact-match 2>/dev/null || true)" = "$TAG" || { echo "HEAD of $ORT_SRC is not $TAG" >&2; exit 1; }

# Source state: the tag, plus the patch when PATCH=1. Tracked files and new
# files under onnxruntime/ are reset; the build folder and emsdk stay.
patch_file="${PATCH_FILE:-$app/reference/onnxruntime-webgpu.patch}"
git reset -q
git checkout -- .
git clean -fdq -- onnxruntime
if [ "$PATCH" = 1 ]; then git apply --whitespace=nowarn "$patch_file"; fi
if [ "$FORCE_DP4A" = 1 ]; then
  [ "$PATCH" = 1 ] || { echo "FORCE_DP4A=1 needs PATCH=1" >&2; exit 1; }
  dp4a_src=onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.cc
  sed -i.bak 's/^#define ORT_FORK_DEBUG_FORCE_DP4A 0$/#define ORT_FORK_DEBUG_FORCE_DP4A 1/' "$dp4a_src" && rm -f "$dp4a_src.bak"
  grep -q '^#define ORT_FORK_DEBUG_FORCE_DP4A 1$' "$dp4a_src" || { echo "could not set ORT_FORK_DEBUG_FORCE_DP4A" >&2; exit 1; }
  echo "DEBUG BUILD: ORT_FORK_DEBUG_FORCE_DP4A=1 (DP4A on every GPU)"
elif [ "$PATCH" = 1 ]; then
  # A normal build must not force DP4A. Catches a patch re-exported from a
  # tree that a FORCE_DP4A=1 build left at 1.
  grep -q '^#define ORT_FORK_DEBUG_FORCE_DP4A 0$' onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.cc || {
    echo "patch does not set ORT_FORK_DEBUG_FORCE_DP4A 0 (a debug patch?); refusing a normal build" >&2; exit 1; }
fi
if [ "$FUSED_DECODE" != 1 ]; then
  [ "$FUSED_DECODE" = 0 ] && [ "$PATCH" = 1 ] || { echo "FUSED_DECODE must be 1 or 0 (0 needs PATCH=1)" >&2; exit 1; }
  fused_src=onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h
  sed -i.bak 's/^#define ORT_FORK_FUSED_DECODE 1$/#define ORT_FORK_FUSED_DECODE 0/' "$fused_src" && rm -f "$fused_src.bak"
  grep -q '^#define ORT_FORK_FUSED_DECODE 0$' "$fused_src" || { echo "could not set ORT_FORK_FUSED_DECODE 0" >&2; exit 1; }
  echo "FUSED_DECODE=0: one-token steps skip the fused MLP/QKV decode programs"
fi
if [ "$PATCH" = 1 ] && [ "$(cd "$(dirname "$patch_file")" && pwd)/$(basename "$patch_file")" = "$app/reference/onnxruntime-webgpu.patch" ]; then
  # The reference patch (however it was named) must carry every K3 part: the
  # row-invariant attention kernel, the q/k norm fold for 1-16 tokens, and the
  # one-workgroup-per-row LayerNorm and SkipLayerNorm. A PATCH_FILE that
  # points at another file (an experiment) skips this check.
  grep -q 'FlashAttentionDecodeGQAProgram' onnxruntime/contrib_ops/webgpu/bert/flash_attention.h &&
    grep -q 'decode_norm_fast_path = has_qk_norm && parameters.sequence_length_ <= 16' onnxruntime/contrib_ops/webgpu/bert/group_query_attention.cc &&
    grep -q 'split_norm_dim = false' onnxruntime/core/providers/webgpu/nn/layer_norm.cc &&
    grep -q 'split_norm_dim = false' onnxruntime/core/providers/webgpu/nn/rms_norm.cc &&
    grep -q 'split_hidden_dim = false' onnxruntime/contrib_ops/webgpu/bert/skip_layer_norm.cc || {
      echo "reference patch lacks the K3 attention/norm changes; refusing" >&2; exit 1; }
  # K4: MatMulNBits for 1-16 rows runs through one pipeline per path (DP4A or
  # float), and 2-16 rows never reach the prefill tile kernels. DP4A keeps the
  # rows in a runtime loop; the float kernel (PK, tile9) has up to 9 register
  # accumulators per tile, one inlined row_sums copy per row.
  grep -q 'const bool batchable = M >= 1 && M <= kMaxBatchedM' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc &&
    grep -q 'ApplyDP4AMatMulNBitsBatched(a, b, scales' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc &&
    grep -q 'if (!small_m && CanApplySubgroupMatrixMatMulNBits' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc &&
    grep -q 'use_wide_tile_program = !small_m &&' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc &&
    grep -q 'DP4AMatMulNBitsBatchedProgram' onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h &&
    grep -q '^#define ORT_FORK_FUSED_DECODE [01]$' onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h &&
    grep -q 'ORT_FORK_FUSED_DECODE != 0 &&' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc &&
    grep -q 'M != 1 || ORT_FORK_FUSED_DECODE == 0' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_qkv.cc &&
    grep -q 'acc8 += row_sums(a_base + 8u \* K_of_a, w,' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_batched.wgsl.template &&
    grep -q 'for (var m = 0u; m < rows; m++) {' onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_batched.wgsl.template || {
      echo "reference patch lacks the K4 batched MatMulNBits kernels; refusing" >&2; exit 1; }
  # tile9k8: 32-bit flash-attention prefill steps 8 keys at a time.
  grep -A1 -F 'if (!is_fp16) {' onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc | grep -q 'return 8;' || {
      echo "reference patch lacks the f32 attention key step 8 (tile9k8); refusing" >&2; exit 1; }
  # K5: CausalConvWithState state window on WebGPU, up to 16 slots.
  grep -q 'constexpr int64_t kMaxStateWindow = 16;' onnxruntime/contrib_ops/cpu/bert/attention_common.h &&
    grep -q 'ParseStateWindow(info, state_window_)' onnxruntime/contrib_ops/webgpu/bert/causal_conv_with_state.cc &&
    grep -q '#if output_present_state && has_state_window' onnxruntime/contrib_ops/webgpu/bert/causal_conv_with_state.wgsl.template || {
      echo "reference patch lacks the K5 CausalConvWithState state window; refusing" >&2; exit 1; }
  # ovis: LinearAttention state window and chunked prefill on WebGPU, and the
  # opt-in tiled fp16 attention prefill (session config webgpu.flash_attention_tiled).
  la=onnxruntime/contrib_ops/webgpu/bert/linear_attention
  grep -q 'ORT_THROW_IF_ERROR(linear_attention_helper::ParseStateWindow(info, state_window_));' $la.cc &&
    grep -q '^#if has_state_window$' $la.wgsl.template &&
    grep -q 'Status LinearAttentionChunkPrepProgram::GenerateShaderCode' $la.cc &&
    grep -q 'Status LinearAttentionChunkScanProgram::GenerateShaderCode' $la.cc &&
    grep -q 'const bool use_chunked = seq_length > 16' $la.cc || {
      echo "reference patch lacks the ovis LinearAttention state window / chunked prefill; refusing" >&2; exit 1; }
  grep -q 'class FlashAttentionTiledProgram final' onnxruntime/contrib_ops/webgpu/bert/flash_attention.h &&
    grep -q 'if (parameters.flash_attention_tiled_ && is_fp16_q && parameters.head_size_ == 64' onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc &&
    grep -q 'GetConfigOrDefault("webgpu.flash_attention_tiled", "0") == "1"' onnxruntime/contrib_ops/webgpu/bert/multihead_attention.cc || {
      echo "reference patch lacks the ovis tiled fp16 attention (webgpu.flash_attention_tiled); refusing" >&2; exit 1; }
  # P5: the fused split-cache attention softmax (com.microsoft SplitMaskedSoftmax),
  # which a gemma12b decoder built with build.py --fuse-softmax needs.
  grep -q 'SplitMaskedSoftmax)>' onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc &&
    grep -q 'SplitMaskedSoftmax, 1,' onnxruntime/core/graph/contrib_ops/bert_defs.cc &&
    grep -q 'Microsoft, 1, SplitMaskedSoftmax)>());' onnxruntime/core/graph/contrib_ops/ms_opset.h || {
      echo "reference patch lacks the P5 SplitMaskedSoftmax op; refusing" >&2; exit 1; }
  # q8: the 8-bit wide-tile prefill loop without the indexed array (14x slow).
  grep -q 'fn b_words(data : BData8, b_idx : u32) -> vec2<u32>' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_wide_tile.wgsl.template || {
      echo "reference patch lacks the 8-bit wide-tile fix (BData8); refusing" >&2; exit 1; }
  # PM: large external initializers go to the GPU in 16 MB pieces with a
  # queue wait every 64 MB (Chrome's staging shared-memory pool stays small).
  grep -q 'webgpuUploadPendingBytes >= 64 \* 1024 \* 1024' onnxruntime/wasm/post-webgpu.js &&
    grep -q 'await Module.webgpuUploadExternalBuffer(dataIdOrBuffer, part, done);' onnxruntime/core/framework/external_data_loader.cc || {
      echo "reference patch lacks the PM chunked weight upload; refusing" >&2; exit 1; }
  # GN (gpu-neutral): every GPU runs the same programs. Kernels read an empty
  # adapter info (no brand, architecture or subgroup width), the device has no
  # subgroup features, and the limits kernels choose from are the WebGPU
  # defaults. Nothing else may read the real adapter.
  ctx=onnxruntime/core/providers/webgpu/webgpu_context
  grep -q 'const wgpu::AdapterInfo& AdapterInfo() const { return kernel_adapter_info_; }' $ctx.h &&
    grep -q 'SplitKConfig>(kernel_adapter_info_)' $ctx.cc &&
    grep -q '^    GpuNeutralLimits(device_limits_);$' $ctx.cc &&
    grep -q 'device_features_.erase(wgpu::FeatureName::Subgroups);' $ctx.cc &&
    grep -q 'device_features_.erase(wgpu::FeatureName::SubgroupSizeControl);' $ctx.cc &&
    grep -q 'device_features_.erase(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);' $ctx.cc &&
    ! sed -n '/GetAvailableRequiredFeatures(const wgpu::Adapter/,/return required_features;/p' $ctx.cc |
        grep -v '^ *//' | grep -q 'Subgroup' &&
    [ "$(grep -cE '(^|[^a-z_])adapter_info_' $ctx.cc)" = 2 ] &&
    ! grep -rn -E 'GetAdapterInfo|\.GetInfo\(|(^|[^a-z_])adapter_info_' onnxruntime/contrib_ops/webgpu onnxruntime/core/providers/webgpu |
        grep -v "^$ctx\." | grep -q . || {
      echo "reference patch is not GPU-neutral (GN): a kernel can see the GPU brand, subgroups or adapter limits; refusing" >&2; exit 1; }
  # F16R (gpu-f16-read): kernels read float16 inputs through 32-bit words.
  # Intel gen-9 on D3D12 returns 0 for the odd element when a kernel picks one
  # f16 out of a 16-bit storage load (docs/LOWMEM.md, Windows GPU). Scalar f16
  # inputs are array<u32>; every f16 getter bitcasts a 32-bit value; Where's
  # broadcast path reads through the getters.
  sv=onnxruntime/core/providers/webgpu/shader_variable
  grep -q 'bool packed_f16_input_ = false;' $sv.h &&
    grep -q '^std::string PackedF16Get(ProgramVariableDataType type' $sv.cc &&
    grep -q 'bitcast<vec4<f16>>(bitcast<vec2<u32>>(' $sv.cc &&
    grep -q 'var->packed_f16_input_ = is_input &&' onnxruntime/core/providers/webgpu/shader_helper.cc &&
    grep -q 'a_input.GetByOffset("index_a" + x)' onnxruntime/core/providers/webgpu/tensor/where.cc || {
      echo "reference patch lacks the F16R float16 read path (gpu-f16-read); refusing" >&2; exit 1; }
  # SLC (gpu-slices): no GPU submit runs long and few wait in the queue, so a
  # slow GPU stays under the Windows 2 s GPU timeout and Chrome can draw the
  # page (docs/LOWMEM.md, Windows GPU). Flash-attention prefill in slices of
  # 2^32 products, each its own submit; matmuls of 2^31+ multiply-adds submit
  # alone; at most 16 submits in flight.
  fa=onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc
  grep -q 'constexpr uint64_t kFlashAttentionSliceWork = uint64_t{1} << 32;' $fa &&
    grep -q '.SetOwnSubmit(slice < total_workgroups);' $fa &&
    grep -q 'constexpr uint64_t kOwnSubmitMatMulWork = uint64_t{1} << 31;' onnxruntime/core/providers/webgpu/program.h &&
    grep -q 'matmul_program.SetOwnSubmit(' onnxruntime/core/providers/webgpu/math/matmul.cc &&
    [ "$(grep -c 'program.SetOwnSubmit(' onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc)" = 2 ] &&
    grep -q 'uint32_t max_submits_in_flight_ = 16;' onnxruntime/core/providers/webgpu/webgpu_context.h &&
    grep -q 'while (submits_in_flight_.size() > max_submits_in_flight_)' onnxruntime/core/providers/webgpu/webgpu_context.cc || {
      echo "reference patch lacks the SLC submit bounds (gpu-slices); refusing" >&2; exit 1; }
fi
git add -N onnxruntime   # so the stat below lists new files too
git diff --stat
git reset -q

( cd cmake/external/emsdk && ./emsdk install "$EMSDK_VERSION" >/dev/null && ./emsdk activate "$EMSDK_VERSION" >/dev/null )

# macOS: /tmp is a symlink to /private/tmp. When the build tree is reached by
# both spellings, emcc's sanity check sees a "changed" config and clears its
# cache (sysroot headers included) while other compiles run, which fails the
# build with missing <cassert>/<initializer_list>. Prime the cache once, then
# skip the check for the parallel build.
em="$ORT_SRC/cmake/external/emsdk/upstream/emscripten"
if [ ! -f "$em/cache/sysroot/include/c++/v1/initializer_list" ]; then
  rm -rf "$em/cache"
  ( cd "$ORT_SRC/cmake/external/emsdk" && source ./emsdk_env.sh >/dev/null 2>&1 && echo 'int main(){}' > /tmp/em-prime.cc && em++ -pthread -msimd128 -c /tmp/em-prime.cc -o /tmp/em-prime.o && rm -f /tmp/em-prime.* )
fi
export EMCC_SKIP_SANITY_CHECK=1

# build.py needs the flatbuffers package; a private venv avoids touching the system Python.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  venv="$ORT_SRC/../ort-venv"
  [ -x "$venv/bin/python3" ] || python3 -m venv "$venv"
  "$venv/bin/python3" -c 'import flatbuffers' 2>/dev/null || "$venv/bin/pip" install -q flatbuffers
  PY="$venv/bin/python3"
fi

common_build_args=(
  --parallel --config Release --skip_submodule_sync
  --build_wasm --enable_wasm_simd --enable_wasm_threads
  --target onnxruntime_webassembly --skip_tests --disable_rtti
  --use_webgpu --use_webnn
  --disable_ml_ops --disable_generation_ops
  --disable_types string float4 float8 optional sparsetensor
  --include_ops_by_config onnxruntime/wasm/reduced_types.config
  --enable_reduced_operator_type_support
  --emsdk_version "$EMSDK_VERSION"
)

asyncify_start=$(date +%s)
# --enable_wasm_api_exception_catching is asyncify-only: build_args.py's
# post-processing (`if args.enable_wasm_api_exception_catching:
# args.disable_wasm_exception_catching = True`) turns it into
# --disable_wasm_exception_catching, and build.py refuses that combination
# for a JSPI build ("Cannot set WebAssembly exception catching in JSPI
# build.", build.py's post-init checks). Confirmed by an actual failed run:
# the first attempt at this JSPI build passed this flag on both builds and
# the JSPI build.py invocation logged exactly that error and exited nonzero
# after printing its parsed Namespace (disable_wasm_exception_catching=True)
# instead of building.
"$PY" tools/ci_build/build.py --build_dir "$BUILD_DIR" --enable_wasm_api_exception_catching "${common_build_args[@]}"
asyncify_seconds=$(( $(date +%s) - asyncify_start ))
echo "asyncify build: ${asyncify_seconds}s"

# The JS API is copied from the app's npm package, so it must be the same
# onnxruntime version as the tag.
npm_version="$(node -p "require('$app/node_modules/onnxruntime-web/package.json').version")"
[ "$npm_version" = "${TAG#v}" ] || { echo "node_modules/onnxruntime-web is $npm_version, not ${TAG#v}" >&2; exit 1; }
mkdir -p "$OUT"
cp "$app"/node_modules/onnxruntime-web/dist/*.mjs "$app"/node_modules/onnxruntime-web/dist/*.wasm "$OUT"/
cp "$BUILD_DIR/Release/ort-wasm-simd-threaded.asyncify.wasm" "$BUILD_DIR/Release/ort-wasm-simd-threaded.asyncify.mjs" "$OUT"/
ls -la "$OUT"/ort-wasm-simd-threaded.asyncify.*

if [ "$JSPI" = 1 ]; then
  # Same source tree and patch, a different cmake build folder (the link
  # flags differ: -s JSPI=1 instead of -s ASYNCIFY=1), producing
  # ort-wasm-simd-threaded.jspi.{wasm,mjs} (cmake/onnxruntime_webassembly.cmake
  # names the target ".jspi" when onnxruntime_ENABLE_WEBASSEMBLY_JSPI=ON).
  # --enable_wasm_jspi needs --disable_wasm_exception_catching unset (see the
  # comment above common_build_args): this invocation omits
  # --enable_wasm_api_exception_catching for exactly that reason, unlike the
  # asyncify build above.
  jspi_start=$(date +%s)
  "$PY" tools/ci_build/build.py --build_dir "$BUILD_DIR_JSPI" --enable_wasm_jspi "${common_build_args[@]}"
  jspi_seconds=$(( $(date +%s) - jspi_start ))
  echo "jspi build: ${jspi_seconds}s"
  cp "$BUILD_DIR_JSPI/Release/ort-wasm-simd-threaded.jspi.wasm" "$BUILD_DIR_JSPI/Release/ort-wasm-simd-threaded.jspi.mjs" "$OUT"/
  ls -la "$OUT"/ort-wasm-simd-threaded.jspi.*
fi
