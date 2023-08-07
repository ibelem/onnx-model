"""
WebNN ops support status for ONNX models.
"""
import argparse
import onnx

def parse_graph(model: str):
    """Parse graph of onnx models"""
    onnx_model = onnx.load(model)
    onnx_ops = []
    for t in onnx_model.graph.node:
        if (t.op_type):
            onnx_ops.append(t.op_type)
    onnx_ops = sorted(list(set(onnx_ops)))
    print(onnx_ops)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="", required=True, type=str)
    args = parser.parse_args()
    print(args.model)
    parse_graph(args.model)

