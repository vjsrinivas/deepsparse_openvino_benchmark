import os
import sys
import numpy as np
import subprocess
from argparse import ArgumentParser
from pathlib import Path
import csv
# subprocess to trigger deepsparse benchmark(?)

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

def represents_int(s:str):
    try: float(s)
    except ValueError: return False
    else: return True

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

proc_output = subprocess.run([
    "deepsparse.benchmark",
    args.model
    ],
    capture_output=True
)

# Parse output to usable data:
output = str(proc_output.stdout, "utf-8")
output = output.split("\n")
output = [out.split(":") for out in output if ":" in out]
output = {out[0].strip():out[1].strip() for out in output}
output = {out:float(val) for out,val in output.items() if represents_int(val)}
mapping_dict = {"Iterations":"real_time", "Latency Mean (ms/batch)":"mean", "Latency Median (ms/batch)":"median", "Latency Std (ms/batch)":"std", "Throughput (items/sec)":"throughput"}
output = {mapping_dict[out]:val for out, val in output.items() if out in mapping_dict}

#with open(os.path.join(output_folder, "{}_baseline_results.csv".format(model_title)), "w") as f:
#    f.write("{},{},{},{},{}".format(real_time, latency_mean, latency_median, latency_std, throughput))

if args.iteration is None:
    filename = "{}_baseline_results.csv".format(model_title)
else:
    filename = "{}_baseline_results_{}.csv".format(model_title, args.iteration)

with open(os.path.join(output_folder, filename), 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(output.keys())
    spamwriter.writerow(output.values())