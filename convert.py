import argparse

import onnx
from onnx_tf.backend import prepare


parser = argparse.ArgumentParser(description="Converts ONNX to Tensorflow.")
parser.add_argument("-o", "--onnx", type=str, help="path to onnx model")
args = parser.parse_args()


def main():
    onnx_model = onnx.load(args.onnx)
    model = prepare(onnx_model)
    model.export_graph("model.pb")


if __name__ == "__main__":
    main()
