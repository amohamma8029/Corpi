"""Authorized temporary model-graph rewrite diagnostic, not model deployment.

Requires the separately approved ONNX/ML-dtypes helper environment. Preserve
verified upstream files, write derived graphs under the owned experiment root,
and compare inference/quality before considering adoption.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from core.search.profiles import profiles, verify_installation


def fuse_projected_attention(model, onnx, heads, hidden):
    """Fuse only a checked attention core, retaining all quantized projections.

    ORT's generic BERT matcher expects float MatMul projections. The pinned E5
    artifact instead has quantized projections, but its resulting Q/K/V tensors
    can feed ORT 1.30's CPU MultiHeadAttention kernel directly. Require the exact
    reshape, transpose, scaling and mask patterns before replacing any nodes.
    """
    import math

    fusions = []
    mask_outputs = {}
    for softmax in list(model.get_nodes_by_op_type("Softmax")):
        attributes = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in softmax.attribute
        }
        if attributes.get("axis", -1) not in (-1, 3):
            continue
        children = model.get_children(softmax)
        if len(children) != 1 or children[0].op_type != "MatMul":
            continue
        context = children[0]
        if context.input[0] != softmax.output[0]:
            continue
        path = model.match_parent_path(softmax, ["Add", "Div", "MatMul"], [0, None, 0])
        if path is None:
            continue
        mask_add, divide, scores = path
        scale = model.get_constant_value(divide.input[1])
        if (
            scale is None
            or scale.size != 1
            or not math.isclose(float(scale.item()), math.sqrt(hidden / heads), rel_tol=1e-6)
        ):
            continue
        query = model.match_parent_path(scores, ["Transpose", "Reshape", "Add"], [0, 0, 0])
        key = model.match_parent_path(scores, ["Transpose", "Reshape", "Add"], [1, 0, 0])
        value = model.match_parent_path(context, ["Transpose", "Reshape", "Add"], [1, 0, 0])
        if any(branch is None for branch in (query, key, value)):
            continue
        branches = (query, key, value)
        permutations = ([0, 2, 1, 3], [0, 2, 3, 1], [0, 2, 1, 3])
        valid = True
        for branch, expected in zip(branches, permutations, strict=True):
            transpose, reshape, _ = branch
            perm = next(
                (list(attr.ints) for attr in transpose.attribute if attr.name == "perm"), None
            )
            shape = model.get_constant_value(reshape.input[1])
            if perm != expected or shape is None or list(shape) != [0, 0, heads, hidden // heads]:
                valid = False
        if not valid:
            continue
        mask = model.match_parent_path(
            mask_add, ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0, 0]
        )
        if mask is None:
            continue
        mask_input = mask[-1].input[0]
        mask_info = model.find_graph_input(mask_input)
        if mask_info is None or len(mask_info.type.tensor_type.shape.dim) != 2:
            continue
        mask_axes = [
            next((list(attr.ints) for attr in node.attribute if attr.name == "axes"), None)
            if len(node.input) == 1
            else model.get_constant_value(node.input[1])
            for node in mask[-2:]
        ]
        if [list(axis) if axis is not None else None for axis in mask_axes] != [[2], [1]]:
            continue
        _, fill = model.get_constant_input(mask[0])
        one = model.get_constant_value(mask[1].input[0])
        if fill is None or fill.size != 1 or float(fill.item()) >= 0:
            continue
        if one is None or one.size != 1 or float(one.item()) != 1:
            continue
        context_children = model.get_children(context)
        if len(context_children) != 1 or context_children[0].op_type != "Transpose":
            continue
        output_transpose = context_children[0]
        perm = next(
            (list(attr.ints) for attr in output_transpose.attribute if attr.name == "perm"), None
        )
        output_children = model.get_children(output_transpose)
        if perm != [0, 2, 1, 3] or len(output_children) != 1:
            continue
        output_reshape = output_children[0]
        if output_reshape.op_type != "Reshape":
            continue
        output_shape = model.get_constant_value(output_reshape.input[1])
        if output_shape is None or list(output_shape) != [0, 0, hidden]:
            continue
        removed = [softmax, mask_add, divide, scores, context, output_transpose, output_reshape]
        removed += [node for branch in branches for node in branch[:2]]
        if not model.is_safe_to_fuse_nodes(
            removed,
            list(output_reshape.output),
            model.input_name_to_nodes(),
            model.output_name_to_node(),
        ):
            continue
        if mask_input not in mask_outputs:
            cast_output = model.create_node_name("corpi_attention_mask_int32")
            model.add_node(
                onnx.helper.make_node(
                    "Cast",
                    [mask_input],
                    [cast_output],
                    name=model.create_node_name("CorpiMaskCast"),
                    to=onnx.TensorProto.INT32,
                )
            )
            mask_outputs[mask_input] = cast_output
        projection_outputs = [branch[-1].output[0] for branch in branches]
        fused = onnx.helper.make_node(
            "MultiHeadAttention",
            [*projection_outputs, "", mask_outputs[mask_input]],
            list(output_reshape.output),
            name=model.create_node_name("CorpiProjectedAttention"),
            domain="com.microsoft",
            num_heads=heads,
            scale=1 / float(scale.item()),
            mask_filter_value=float(fill.item()),
        )
        model.remove_nodes(removed)
        model.add_node(fused)
        fusions.append(
            {
                "node": fused.name,
                "projection_inputs": projection_outputs,
                "mask": mask_input,
                "scale": 1 / float(scale.item()),
                "mask_filter_value": float(fill.item()),
            }
        )
    if not fusions:
        raise ValueError("no checked projected-attention patterns matched")
    model.prune_graph()
    return fusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--profile", choices=("minilm-en", "multilingual-e5"), required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--projected-attention", action="store_true")
    parser.add_argument(
        "--static-shapes-only",
        action="store_true",
        help="Use supplied graph shape metadata without symbolic-inference helpers",
    )
    args = parser.parse_args()
    onnx = import_module("onnx")
    transformers = import_module("onnxruntime.transformers")

    # ORT 1.30's installed rewrite helpers use sibling absolute imports.
    if transformers.__file__ is None:
        raise RuntimeError("installed transformer helper location unavailable")
    helper_path = Path(transformers.__file__).parent
    sys.path.insert(0, str(helper_path))
    FusionOptions = import_module("fusion_options").FusionOptions
    BertOnnxModel = import_module("onnx_model_bert").BertOnnxModel

    profile = profiles()[args.profile]
    installed = args.models_directory / profile.identity
    verify_installation(installed, profile)
    config = json.loads((installed / "config.json").read_text("utf-8"))
    model_file = installed / profile.model_file
    raw = model_file.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    started = time.perf_counter()
    graph = onnx.load_model(str(model_file))
    before = Counter(node.op_type for node in graph.graph.node)
    initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    dequantization = []
    for node in graph.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        consumers = [other for other in graph.graph.node if node.output[0] in other.input]
        dequantization.append(
            {
                "node": node.name,
                "inputs": [
                    {"name": name, "shape": list(initializers[name].dims)}
                    for name in node.input
                    if name in initializers
                ],
                "consumers": [
                    {
                        "name": other.name,
                        "op": other.op_type,
                        "inputs": list(other.input),
                        "outputs": list(other.output),
                    }
                    for other in consumers
                ],
            }
        )
    if args.inspect_only:
        result = {
            "profile": profile.id,
            "source_sha256": source_hash,
            "dequantization": dequantization,
        }
        (args.directory / f"{profile.id}-graph-inspection.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return
    optimizer = BertOnnxModel(graph, config["num_attention_heads"], config["hidden_size"])
    if args.static_shapes_only:
        optimizer.disable_shape_inference()
    options = FusionOptions("bert")
    options.enable_gelu_approximation = False
    # The pinned export uses opset 11. ORT's Python optimizer emits default-domain
    # LayerNormalization (standardized only at opset 17); leave that fusion to
    # the native runtime instead, keeping the saved graph checker-valid.
    options.enable_layer_norm = False
    options.enable_skip_layer_norm = False
    optimizer.optimize(options)
    attention_fusions = []
    if args.projected_attention:
        attention_fusions = fuse_projected_attention(
            optimizer, onnx, config["num_attention_heads"], config["hidden_size"]
        )
        if len(attention_fusions) != config["num_hidden_layers"]:
            raise ValueError("not every configured attention layer matched")
    optimizer.topological_sort()
    variant = "projected-attention-checked" if args.projected_attention else "specialized-checked"
    target = args.directory / f"{profile.id}-{variant}.onnx"
    if target.exists():
        raise ValueError("derived graph target already exists")
    if not any(item.domain == "com.microsoft" for item in optimizer.model.opset_import):
        optimizer.model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(optimizer.model)
    optimizer.save_model_to_file(str(target))
    after = Counter(node.op_type for node in optimizer.model.graph.node)
    result = {
        "scope": "temporary specialized graph rewrite; not deployed or accepted",
        "profile": profile.id,
        "manifest": profile.identity,
        "source_sha256": source_hash,
        "derived_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "derived_bytes": target.stat().st_size,
        "rewrite_seconds": time.perf_counter() - started,
        "onnx": version("onnx"),
        "onnxruntime": version("onnxruntime"),
        "head_count": config["num_attention_heads"],
        "hidden_size": config["hidden_size"],
        "before_operators": dict(before),
        "after_operators": dict(after),
        "fused_statistics": optimizer.get_fused_operator_statistics(),
        "gelu_approximation": False,
        "layer_norm_rewrite": False,
        "symbolic_shape_inference": not args.static_shapes_only,
        "projected_attention_fusions": attention_fusions,
    }
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == source_hash
    receipt = "projected-attention-rewrite" if args.projected_attention else "rewrite"
    (args.directory / f"{profile.id}-{receipt}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
