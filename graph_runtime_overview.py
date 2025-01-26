import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=False, default="cpus")
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
    CPUS = os.listdir(args.root)
    ROOTS = [os.path.join(args.root, cpu) for cpu in CPUS]

    # clean up CPU title:
    #CPU = CPU.replace("processor_", "").replace("_processor", "")
    #ROOT = os.path.join(args.root, "output")
    OUTPUT_PATH = args.output
    CATEGORIES = [
                    "baseline",
                    "deepsparse",
                    "openvino",
                    "ort"
                ]
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    all_data = {cat:{_cpu:[] for _cpu in CPUS} for cat in CATEGORIES}
    all_data_ranges = {cat:[{_cpu:[] for _cpu in CPUS}, {_cpu:[] for _cpu in CPUS}] for cat in CATEGORIES}
    
    category2title = {
                        "baseline": "DeepSparse (deepsparse.benchmark)",
                        "deepsparse": "DeepSparse",
                        "openvino": "OpenVINO",
                        "ort": "ONNXRuntime (Baseline)"
                    }
    cpu2title = {
                    "intel_xeon_platinum_8175m_cpu_2":"t3.xlarge",
                    "intel_xeon_e5-2666_v3_cpu":"c4.xlarge",
                    "amd_epyc_7r13_cpu":"m6a.xlarge",
                    "intel_xeon_platinum_8175m_cpu":"m5.xlarge",
                    "intel_xeon_platinum_8259cl_cpu":"m5n.xlarge",
                    "intel_xeon_platinum_8488c_cpu":"m7i.xlarge"

                    #"amd_epyc_7r13_cpu":"AMD EPYC 7R13",
                    #"intel_xeon_e5-2666_v3_cpu":"Intel Skylake\nE5 2686 v3",
                    #"intel_xeon_platinum_8175m_cpu":"Intel Xeon\nPlatinum 8175",
                    #"intel_xeon_platinum_8175m_cpu_2":"Intel Xeon\nPlatinum 8175",
                    #"intel_xeon_platinum_8259cl_cpu":"Intel Xeon\nPlatinum 8259",
                    #"intel_xeon_platinum_8488c_cpu":"Intel Xeon\nScalable 8488C"

                    #"amd_ryzen_7_2700x_eight-core_processor":"AMD Ryzen 7 2700X",
                    #"amd_ryzen_7_2700x_eight-core_processor_2":"AMD Ryzen 7 2700X 2",
                    #"amd_ryzen_7_2700x_eight-core_processor_3":"AMD Ryzen 7 2700X 3",
                    #"amd_ryzen_7_2700x_eight-core_processor_4":"AMD Ryzen 7 2700X 4"
                }

    for cpu_id, root in enumerate(ROOTS):
        for folder in os.listdir(os.path.join(root)):
        #for folder in os.listdir(os.path.join(root, "output")):
            for category in CATEGORIES:
                csv_files = os.listdir(os.path.join(root, folder))
                csv_files = [_csv for _csv in csv_files if category in _csv]
                
                _data = []
                for file in csv_files:
                    _file = os.path.join(root, folder, file)
                    data = pd.read_csv(_file)
                    data = data.rename(columns=lambda x: x.strip())
                    _data.append(data["throughput"].values[0])
                _max_data = np.max(_data)
                _min_data = np.min(_data)
                _data = np.median(_data)
                all_data[category][CPUS[cpu_id]].append(_data)
                all_data_ranges[category][0][CPUS[cpu_id]].append(_max_data)
                all_data_ranges[category][1][CPUS[cpu_id]].append(_min_data)

                #print(_data, category, folder)

    fig, ax = plt.subplots(1,1, figsize=(10,6))
    n_cpus = [i for i in range(len(CPUS))]
    markers = ["^", "s", "o", "D"]
    assert len(markers) == len(all_data)
    
    for r, runtime in enumerate(all_data.keys()):
        _cpus = []
        _cpus_max = []
        _cpus_min = []
        #for cpu in all_data[runtime].keys():
        for c, cpu in enumerate(cpu2title.keys()):
            all_models_data = all_data[runtime][cpu]
            model_data = np.mean(all_models_data)
            model_data_max = np.mean(all_data_ranges[runtime][0][cpu])
            model_data_min = np.mean(all_data_ranges[runtime][1][cpu])
            _cpus.append(model_data)
            _cpus_max.append(model_data_max)
            _cpus_min.append(model_data_min)
        ax.fill_between(n_cpus, y1=_cpus_max, y2=_cpus_min, alpha=0.15)
        print(">>>", category2title[runtime], _cpus)
        ax.plot(n_cpus, _cpus, label=category2title[runtime], marker=markers[r])

    ax.set_xticks(n_cpus)
    ax.set_xticklabels(list(cpu2title.values()))
    ax.set_ylabel("Throughput (imgs/sec)")
    ax.set_xlabel("AWS Machine Type")
    ax.set_title("Average Inference Engine Throughput vs AWS Instances")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(args.output, "overview_line_graph.png"))
    plt.savefig(os.path.join(args.output, "overview_line_graph.svg"))