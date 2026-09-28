"""Windows GPU plan: a copy of Small's vision graph with extra outputs ("taps").

Every node output in the patch embedder and in layer 0, and every output of the
MultiHeadAttention and SkipSimplifiedLayerNormalization nodes of all layers,
becomes a graph output. The weights stay in the package's vision.onnx_data
(same external-data location), so only the graph file is new.

  python tools/gpu-debug/vision_taps.py <package dir> <out.onnx>

gpu-check.html?taps=1 runs it (src/gpu-check/arm.js) and posts the first rows
of every tap, so an M5 run and a slow-GPU run can be compared op by op.
"""
import sys
import onnx
from onnx import helper, shape_inference

pkg, out = sys.argv[1], sys.argv[2]
m = onnx.load(f'{pkg}/vision.onnx', load_external_data=False)
try:
    inferred = shape_inference.infer_shapes(m, strict_mode=False)
except Exception:
    inferred = m
types = {vi.name: vi.type.tensor_type.elem_type for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input)}
for init in m.graph.initializer:
    types[init.name] = init.data_type

# Walk the graph in order: a Cast gives its 'to' type, a shape-like op gives
# int64, anything else the type of its first input. Inferred types win.
SHAPE_OPS = {'Shape': onnx.TensorProto.INT64, 'Equal': onnx.TensorProto.BOOL, 'Not': onnx.TensorProto.BOOL}
for node in m.graph.node:
    for index, name in enumerate(node.output):
        if not name or types.get(name):
            continue
        if node.op_type == 'Cast':
            types[name] = next(a.i for a in node.attribute if a.name == 'to')
        elif node.op_type in SHAPE_OPS:
            types[name] = SHAPE_OPS[node.op_type]
        elif node.op_type == 'SkipSimplifiedLayerNormalization' and index in (1, 2):
            types[name] = onnx.TensorProto.FLOAT
        else:
            first = next((i for i in node.input if i), None)
            types[name] = types.get(first) or onnx.TensorProto.FLOAT16

def type_of(node, name):
    return types[name]

existing = {o.name for o in m.graph.output}
taps = []
for node in m.graph.node:
    # Index, shape and weight-layout ops carry no arithmetic worth a tap,
    # except in the patch embedder (the position-table lookups).
    if node.op_type in ('Constant', 'Shape') or (node.op_type in ('Transpose', 'Unsqueeze', 'Gather', 'Equal', 'Not', 'Reshape', 'Where', 'Slice') and 'patch_embedder' not in node.name):
        continue
    early = 'patch_embedder' in node.name or 'layers.0/' in node.name or 'layers.0.' in (node.output[0] if node.output else '')
    norm = node.op_type in ('SkipSimplifiedLayerNormalization', 'MultiHeadAttention')
    if not (early or norm):
        continue
    for index, name in enumerate(node.output):
        # SkipSimplifiedLayerNormalization outputs 1 and 2 are fp32 statistics.
        if node.op_type == 'SkipSimplifiedLayerNormalization' and index in (1, 2):
            continue
        if name and name not in existing and name not in taps:
            taps.append(name)
            m.graph.output.append(helper.make_tensor_value_info(name, type_of(node, name), None))
onnx.save(m, out)
print(f'{len(taps)} taps -> {out}')
