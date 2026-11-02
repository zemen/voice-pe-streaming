"""Regression: the process-wide ORT single-thread patch must not break positional sess_options."""

import onnxruntime as ort

import wyoming_livekit_speechbrain.__main__  # noqa: F401  (installs the InferenceSession patch)


def _tiny_model(path):
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def test_positional_and_keyword_sess_options(tmp_path):
    p = str(tmp_path / "m.onnx")
    _tiny_model(p)
    so = ort.SessionOptions()
    ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])
    ort.InferenceSession(p, providers=["CPUExecutionProvider"])
