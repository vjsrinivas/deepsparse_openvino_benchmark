import os
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default="../amd_ryzen_7_2700x_eight-core_processor")
    parser.add_argument("--output", type=str, default="graphs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    runtime_mapping = {"baseline": "ONNXRuntime", "deepsparse": "DeepSparse", "openvino": "OpenVINO"}
    fig, ax = plt.subplots(len(runtime_mapping), 1, figsize=(5,10))

    for i, (runtime, runtime_mapped) in enumerate(runtime_mapping.items()):
        prune_mapping = {"output_default_prune": "Default Sparsity", "output_85_prune": "85% Sparsity", "output_99_prune": "99% Sparsity"}
        model_mapping = {"mobilevit_xs.cvnets_in1k":"MobileViT-xs", "tf_efficientnet_b0.in1k":"EfficientNet-b0", "tf_efficientnet_b2.in1k":"EfficientNet-b2"}
        data = {_model:{_id:0.0 for _id in prune_mapping.values()} for _model in model_mapping.values()}

        for _folder in os.listdir(args.source):
            __root = os.path.join(args.source, _folder)
            for _sub_folder in os.listdir( __root ):
                _model_base = "_".join(_sub_folder.split("_")[:-1])
                #_model_base = _sub_folder
                _model = model_mapping[_model_base]
                __subroot = os.path.join(__root, _sub_folder)
                for _file in os.listdir(__subroot):
                    file_path = os.path.join(__subroot, _file)
                    if "results_0" in file_path and runtime in file_path:
                        _csv_data = pd.read_csv(file_path, skipinitialspace=True)
                        throughput = _csv_data["throughput"][0]
                        _folder_mapped = prune_mapping[_folder]
                        data[_model][_folder_mapped] = throughput

        #plt.figure(figsize=(9,6))
        for _network in model_mapping.values():
            print(_network)
            ax[i].plot([ data[_network][_prune_key] for _prune_key in prune_mapping.values() ], "-o", label=_network)
        ax[i].set_title("{} Sparsity Testing".format(runtime_mapped))
        ax[i].set_ylabel("Throughput (img/s)")
        ax[i].set_xlabel("Sparsity")
        ax[i].set_xticks([i for i in range(len(prune_mapping))], [ _prune_key for _prune_key in prune_mapping.values() ] )
        #plt.x([ _prune_key for _prune_key in prune_mapping.values() ])
        ax[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "fake_prune_throughput.svg"))