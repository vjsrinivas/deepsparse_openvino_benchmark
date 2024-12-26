import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, default="graphs", required=False)
    return parser.parse_args()

def create_clustered_horizontal_bar_chart(categories, data, category_labels, title, colors=None) -> tuple:
    # Number of categories
    n_categories = len(categories)
    n_groups = len(data)
    bar_width = 0.8 / n_groups
    index = np.arange(n_categories)

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_groups))

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bars for each group in each category
    rectified_cat_labels = [label.replace("imagenet-", "").replace("imagenet_", "").replace("-imagenet", "") for label in categories]
    rectified_cat_labels = ["{}pruned".format(label.split("pruned")[0]) if "pruned" in label else label for label in rectified_cat_labels]

    for i in range(n_groups):
        print(index + i * bar_width)
        point_arrays = index + i * bar_width
        ax.barh(point_arrays, data[i], bar_width, label=category_labels[i], color=colors[i])
        #ax.axhline(y=i - 0.8, color='k', linestyle='--', linewidth=2)

    for i, pa in enumerate(point_arrays):
        if i % 2 == 0 and i != 0:
            ax.axhline(y=pa-0.8, color='k', linestyle='--', linewidth=1, alpha=0.5)  # Example line
        
    # Set the labels for the y-axis and x-axis
    ax.set_yticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_yticklabels(rectified_cat_labels)
    ax.set_xlabel('Throughput (imgs/sec)')
    
    # Set the title and display the legend
    ax.set_title(title)
    #ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    

    # Display the plot
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    args = parse_args()
    CPU = Path(args.root).stem 
    # clean up CPU title:
    CPU = CPU.replace("processor_", "").replace("_processor", "")
    #ROOT = os.path.join(args.root, "output")
    ROOT = args.root
    OUTPUT_PATH = args.output
    CATEGORIES = ["baseline", "deepsparse", "openvino", "ort"]
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    all_data = {}
    inverse_all_data = {}
    category2title = {"baseline": "DeepSparse (deepsparse.benchmark)", "deepsparse": "DeepSparse", "openvino": "OpenVINO", "ort": "ONNXRuntime (Baseline)"}

    for folder in os.listdir(ROOT):
        run_data_throughput = {}
        _full_folder = os.path.join(ROOT, folder)

        for category in CATEGORIES:
            _data = []
            csv_files = os.listdir(_full_folder)
            csv_files = [_csv for _csv in csv_files if category in _csv]
            iterations = [int(_csv.split("_")[-1].split('.')[0]) for _csv in csv_files]
            
            for file in csv_files:
                _file = os.path.join(_full_folder, file)
                data = pd.read_csv(_file)
                data = data.rename(columns=lambda x: x.strip())
                _data.append(data["throughput"].values[0])
            _data = np.median(_data)

            category_title = category2title[category]
            run_data_throughput[category_title] = _data

        all_data[folder] = run_data_throughput

    # Iterate over the outer dictionary
    for outer_key, inner_dict in all_data.items():
        # Iterate over the inner dictionary
        for inner_key, inner_value in inner_dict.items():
            # If the inner key already exists in the inverted dictionary, append the new value
            if inner_key not in inverse_all_data:
                inverse_all_data[inner_key] = {}
            inverse_all_data[inner_key][outer_key] = inner_value

    sorted_keys = sorted(list(all_data.keys()))
    #sorted_runtime_keys = sorted(list(inverse_all_data.keys()))
    sorted_runtime_keys = list(category2title.values())
    struct_runtimes = [[] for _ in sorted_runtime_keys]
    sorted_keys2idx = {k:i for i,k in enumerate(sorted_runtime_keys)}

    for _key in sorted_runtime_keys:
        for _inner_key in sorted_keys:
            struct_runtimes[sorted_keys2idx[_key]].append(inverse_all_data[_key][_inner_key])

    #print(all_data)
    #print()
    #for i, srt in enumerate(struct_runtimes):
    #    print(sorted_runtime_keys[i], srt)
    #exit()

    categories = sorted_keys
    data = struct_runtimes
    category_labels = sorted_runtime_keys
    fig, ax = create_clustered_horizontal_bar_chart(categories, data, category_labels, "Network Throughput vs Runtimes\n(CPU - {})".format(CPU))
    plt.savefig(os.path.join(args.output, "overview_graph_{}.png".format(CPU)))
    plt.savefig(os.path.join(args.output, "overview_graph_{}.svg".format(CPU)))