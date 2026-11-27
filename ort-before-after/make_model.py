"""Builds the two test models of this page and their expected outputs.

  pip install onnx numpy
  python make_model.py

Writes gather_f16.onnx, cast_f16.onnx and expected.json next to this file.

gather_f16.onnx: Gather over a float16 table [64, 16] (an initializer, like an
embedding table). The row indices are a graph input, so the Gather runs on the
GPU and is not folded away when the model loads. Table value k (row-major) is
(k + 1) / 4: every value is distinct, nonzero and exact in float16.

cast_f16.onnx (a control): Cast a float16 input [1023] to float32, then add 0.
ONNX Runtime's Cast kernel reads four float16 values at a time (vec4<f16>),
not one, so this model is not expected to show the bug.

Expected outputs are computed here on the CPU with numpy.
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
ROWS, COLS = 64, 16
CAST_LEN = 1023
OPSET, IR_VERSION = 17, 8


def save(graph, name):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="axiya-ort-before-after")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, HERE / name)


# Gather over a float16 table.
table = ((np.arange(ROWS * COLS, dtype=np.float32) + 1) / 4).astype(np.float16).reshape(ROWS, COLS)
assert len(np.unique(table)) == table.size and np.all(table != 0)
indices = np.random.default_rng(16).permutation(ROWS).astype(np.int64)
save(helper.make_graph(
    [helper.make_node("Gather", ["table", "indices"], ["out"], axis=0)],
    "gather_f16",
    [helper.make_tensor_value_info("indices", TensorProto.INT64, [ROWS])],
    [helper.make_tensor_value_info("out", TensorProto.FLOAT16, [ROWS, COLS])],
    [numpy_helper.from_array(table, "table")],
), "gather_f16.onnx")
gather_out = table[indices]
# Position of each output value inside the table (row-major): its parity is
# the parity of the 16-bit element the GPU had to pick out.
gather_offset = (indices[:, None] * COLS + np.arange(COLS)[None, :]).reshape(-1)

# Cast float16 -> float32, then add 0.
x = ((np.arange(CAST_LEN, dtype=np.float32) + 1) / 4).astype(np.float16)
save(helper.make_graph(
    [helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
     helper.make_node("Add", ["xf", "zero"], ["y"])],
    "cast_f16",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [CAST_LEN])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [CAST_LEN])],
    [numpy_helper.from_array(np.zeros([1], np.float32), "zero")],
), "cast_f16.onnx")
cast_out = x.astype(np.float32) + np.float32(0)

(HERE / "expected.json").write_text(json.dumps({
    "note": "Made by make_model.py with numpy on the CPU. Values are compared bit for bit.",
    "gather": {
        "model": "gather_f16.onnx",
        "indices": indices.tolist(),
        "shape": [ROWS, COLS],
        "expected_f16_bits": gather_out.reshape(-1).view(np.uint16).tolist(),
        "source_offset": gather_offset.tolist(),
    },
    "cast": {
        "model": "cast_f16.onnx",
        "input_f16_bits": x.view(np.uint16).tolist(),
        "expected_f32": cast_out.tolist(),
        "source_offset": list(range(CAST_LEN)),
    },
}))
print("gather", gather_out.shape, "cast", cast_out.shape)
