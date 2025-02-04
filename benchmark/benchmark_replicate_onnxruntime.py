import onnxruntime as ort
import time
import numpy as np
import subprocess
from tqdm import tqdm
from utils import get_input_output
from argparse import ArgumentParser
import os
from pathlib import Path

# Get CPU model on Linux
def get_cpu_name():
    try:
        # Run the `lscpu` command to get CPU info
        result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Parse the output to get the CPU model name
        for line in result.stdout.decode().splitlines():
            if line.startswith('Model name'):
                return line.split(':')[1].strip()
    except Exception as e:
        return f"Error: {e}"
    
parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--iteration", type=int, required=False, default=None)
args = parser.parse_args()

# get cpu data and add that to the output
cpu_name = get_cpu_name()
cpu_name = cpu_name.lower().replace(" ", "_")

model_title = str(Path(args.model).stem)
output_folder = os.path.join(cpu_name, "output", model_title)
os.makedirs(output_folder, exist_ok=True)
assert os.path.exists(args.model)

# download onnx, compile
model_name = args.model
input_dims, output_dims = get_input_output(model_name)
model = ort.InferenceSession(model_name, providers=["CPUExecutionProvider"])

# run inference (input is raw numpy tensors, output is raw scores)
latency = []
length = 10 # in seconds
start_time = time.time()
end_time = time.time()
passes = 0
experiment_length = 60
stream_end_time = time.perf_counter() + experiment_length

inputs = {_key:np.random.normal(0, 2.5, size=tuple(_dim)).astype(np.float32) for _key, _dim in input_dims.items()} 
output_keys = list(output_dims.keys()) 

t3 = time.time()
while time.perf_counter() < stream_end_time:
    t1 =  time.perf_counter()
    output = model.run(output_keys, inputs)
    t2 = time.perf_counter()
    latency.append(t2-t1)
    passes += 1
t4 = time.time()

latency_mean = np.mean(latency)*1000.0
latency_median = np.median(latency)*1000.0
latency_std = np.std(latency)*1000.0
throughput = passes/experiment_length
real_time = t4-t3

if args.iteration is None:
    filename = "{}_ort_results.csv".format(model_title)
else:
    filename = "{}_ort_results_{}.csv".format(model_title, args.iteration)

print("real time (s): {} | mean (ms): {} | median (ms): {} | std (ms): {} | throughput (elements/sec): {}".format(real_time, latency_mean, latency_median, latency_std, throughput))

with open(os.path.join(output_folder, filename), "w") as f:
    f.write("real_time, mean, median, std, throughput\n")
    f.write("{},{},{},{},{}".format(real_time, latency_mean, latency_median, latency_std, throughput))