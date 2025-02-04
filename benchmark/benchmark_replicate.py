from deepsparse import Engine
import time
import numpy as np
import subprocess
from tqdm import tqdm
import os
from argparse import ArgumentParser
from pathlib import Path

from deepsparse.cpu import cpu_architecture
from deepsparse.benchmark.helpers import (
    decide_thread_pinning,
    parse_num_streams,
    parse_scenario,
    parse_scheduler,
)
from utils import get_input_output

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

thread_pinning = "core" # none, core, numa
num_cores = cpu_architecture().num_available_physical_cores
decide_thread_pinning(thread_pinning)
scenario = "sync"
scenario = parse_scenario(scenario.lower())
scheduler = parse_scheduler(scenario)
num_streams = parse_num_streams(1, num_cores, scenario)

# ignore kv caching because we're focusing only on CNNs for now

# download onnx, compile
#_zoo_stub = Path(args.model)
#zoo_stub="zoo:{}".format(_zoo_stub.stem)
#inferred_model = os.path.join("models", "{}.onnx".format(zoo_stub.split(":")[1]))
input_dims, output_dims = get_input_output(args.model)
compiled_model = Engine(
                            model=args.model,
                            batch_size=1,
                            num_cores=num_cores,
                            num_streams=num_streams,
                            scheduler=scheduler
                        )

# run inference (input is raw numpy tensors, output is raw scores)
latency = []
start_time = time.time()
end_time = time.time()
passes = 0
experiment_length = 60
stream_end_time = time.perf_counter() + experiment_length

#inputs = [np.random.normal(0, 2.5, size=(1,3,224,224)).astype(np.float32)]
inputs = [np.random.normal(0, 2.5, size=_item).astype(np.float32) for _item in input_dims.values()]

t3 = time.time()
while time.perf_counter() < stream_end_time:
    t1 = time.perf_counter()
    output = compiled_model(inputs)
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
    filename = "{}_deepsparse_results.csv".format(model_title)
else:
    filename = "{}_deepsparse_results_{}.csv".format(model_title, args.iteration)

print("real time (s): {} | mean (ms): {} | median (ms): {} | std (ms): {} | throughput (elements/sec): {}".format(real_time, latency_mean, latency_median, latency_std, throughput))

with open(os.path.join(output_folder, filename), "w") as f:
    f.write("real_time, mean, median, std, throughput\n")
    f.write("{},{},{},{},{}".format(real_time, latency_mean, latency_median, latency_std, throughput))