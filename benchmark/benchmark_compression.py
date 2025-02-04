import os
import sys
import gzip
from pathlib import Path


if __name__ == "__main__":
    model_path = "/home/vijay/Documents/devmk6/neuralmagic/deepsparse/models"
    compression_level = 9

    #data = {}
    out_data = open("../pruning_exp_compression_results/compression_results.csv", "w")
    out_data.write("filename,mb\n")

    for model in os.listdir(model_path):
        _model_path = os.path.join(model_path, model)
        for onnx_file in os.listdir(_model_path):
            onnx_file_stem = Path(onnx_file).stem
            pull_path = os.path.join(_model_path, onnx_file)
            print(pull_path)

            data = open(pull_path, "rb").read()
            with open("/tmp/compressed_file.gz", "wb") as f:
                # Create a gzip compressor object with the specified compression level
                with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=compression_level) as gz:
                    # Write the data to the gzip compressor
                    gz.write(data)

            file_size = os.path.getsize("/tmp/compressed_file.gz")
            file_size_in_mb = file_size/1000000
            print("size: {:0.2f}".format(file_size_in_mb))
            #data[onnx_file_stem] = file_size
            out_data.write("{},{:0.2f}\n".format(onnx_file_stem, file_size_in_mb))
        print("===========================================")

    