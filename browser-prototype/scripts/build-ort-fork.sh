#!/usr/bin/env bash
# Builds onnxruntime-web's WebGPU wasm (ort-wasm-simd-threaded.asyncify.{wasm,mjs})
# from the onnxruntime v1.30.0 tag, with or without the batched-int4 patch.
#
#   ORT_SRC=/path/to/ort-fork  PATCH=1  OUT=/path/to/out  scripts/build-ort-fork.sh
#
# Env:
#   ORT_SRC  checkout folder (cloned here if missing; shallow, with shallow submodules)
#   PATCH    1 = apply reference/onnxruntime-webgpu.patch (default), 0 = unmodified tag
#   OUT      folder that receives a full onnxruntime-web dist: the npm 1.30.0 dist
#            files, with the two asyncify files replaced by this build
#   BUILD_DIR  cmake build folder (default $ORT_SRC/build/wasm_webgpu)
#   PATCH_FILE patch to apply when PATCH=1 (default reference/onnxruntime-webgpu.patch)
#   FORCE_DP4A 1 = debug build: the DP4A MatMulNBits path also runs on Apple GPUs
#            (sets ORT_FORK_DEBUG_FORCE_DP4A in dp4a_matmul_nbits.cc), so a Mac
#            runs the int8-activation arithmetic of Intel/AMD/NVIDIA GPUs. Needs
#            PATCH=1. Test builds only; never ship one. Default 0.
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
OUT="${OUT:?set OUT to the output dist folder}"
BUILD_DIR="${BUILD_DIR:-$ORT_SRC/build/wasm_webgpu}"

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

"$PY" tools/ci_build/build.py \
  --build_dir "$BUILD_DIR" \
  --parallel --config Release --skip_submodule_sync \
  --build_wasm --enable_wasm_simd --enable_wasm_threads \
  --target onnxruntime_webassembly --skip_tests --disable_rtti \
  --use_webgpu --use_webnn \
  --enable_wasm_api_exception_catching \
  --disable_ml_ops --disable_generation_ops \
  --disable_types string float4 float8 optional sparsetensor \
  --include_ops_by_config onnxruntime/wasm/reduced_types.config \
  --enable_reduced_operator_type_support \
  --emsdk_version "$EMSDK_VERSION"

# The JS API is copied from the app's npm package, so it must be the same
# onnxruntime version as the tag.
npm_version="$(node -p "require('$app/node_modules/onnxruntime-web/package.json').version")"
[ "$npm_version" = "${TAG#v}" ] || { echo "node_modules/onnxruntime-web is $npm_version, not ${TAG#v}" >&2; exit 1; }
mkdir -p "$OUT"
cp "$app"/node_modules/onnxruntime-web/dist/*.mjs "$app"/node_modules/onnxruntime-web/dist/*.wasm "$OUT"/
cp "$BUILD_DIR/Release/ort-wasm-simd-threaded.asyncify.wasm" "$BUILD_DIR/Release/ort-wasm-simd-threaded.asyncify.mjs" "$OUT"/
ls -la "$OUT"/ort-wasm-simd-threaded.asyncify.*
