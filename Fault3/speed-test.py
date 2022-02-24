import math, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from models import DAVE2pytorch
from DatasetGenerator import DataSequence
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from ast import literal_eval

def intake_racetrack(parentdir="beamng-industrial-racetrack"):
    import shlex
    with open(Path(f"{os.getcwd()}/../{parentdir}/adjusted-middle.txt"),'r') as f:
        lines = f.readlines()
        roadmiddle = [literal_eval(','.join(shlex.split(arr))) for arr in lines]
    with open(Path(f"{os.getcwd()}/../{parentdir}/road-left.txt")) as f:
        lines = f.readlines()
        roadleft = [literal_eval(arr) for arr in lines]
    with open(Path(f"{os.getcwd()}/../{parentdir}/road-right.txt")) as f:
        lines = f.readlines()
        roadright = [literal_eval(arr) for arr in lines]
    return roadmiddle, roadleft, roadright

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("testsetpath", type=Path)
    return parser.parse_args()

args = parse_args()
device = torch.device('cpu')
modelpath = "model-DAVE2PytorchModel-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
model = torch.load(modelpath, map_location=device)
print(model)
dataset = DataSequence(args.testsetpath, transform=Compose([ToTensor()]))
roadmiddle, roadleft, roadright = intake_racetrack()

print("Retrieving output distribution....")
# print("Moments of distribution:", dataset.get_outputs_distribution())
print("Total samples:", dataset.get_total_samples())

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

BATCH_SIZE=1
testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=worker_init_fn)
failure_point = [150, -175]
testdir = "testoutput"
if not os.path.exists(testdir):
    os.mkdir(testdir)
okay_points = []
error_points = []
error_threshold = 0.33
for i, hashmap in enumerate(testloader, 0):
    sample_num = int(hashmap['filename'][0].strip("sample-").strip(".jpg"))
    position = literal_eval(hashmap['position'][0].replace("  ",","))
    # sample_num = int(re.search(r'\d+', sample_num).group())
    pos = [float(s) for s in hashmap['position'][0][1:-1].split("  ")]
    x = hashmap['image'].float().to(device)
    y = hashmap['steering_input'].float().to(device)
    outputs = model(x)
    if abs(outputs.item() - y.item()) > error_threshold:
        error_points.append(pos)
    else:
        okay_points.append(pos)
    dist_to_fail = math.sqrt(math.pow(pos[0] - failure_point[0], 2) + math.pow(pos[1] - failure_point[1], 2))
    if dist_to_fail < 10:
        print(f"{hashmap['filename'][0]}\tcorrect output={y.item():.3f}\tmodel output={outputs.item():.3f}\t\terror={outputs.item()-y.item():.3f}")
        if round(abs(outputs.item() - y.item()), 1) >= error_threshold:
            plt.title(f"error={outputs.item()-y.item():.3f}", color="red")
        else:
            plt.title(f"error={outputs.item() - y.item():.3f}", color="black")
        print("pos={}\n".format(pos))
        plt.imshow(x[0].permute(1,2,0))
        plt.savefig(f"{testdir}/sample-{sample_num}-error.jpg")

plt.close("all")
plt.plot([i[0] for i in roadleft], [i[1] for i in roadleft], "k")
plt.plot([i[0] for i in roadright], [i[1] for i in roadright], "k")
plt.plot([i[0] for i in roadmiddle], [i[1] for i in roadmiddle], "k")
plt.plot([i[0] for i in okay_points], [i[1] for i in okay_points], "b")
plt.scatter([i[0] for i in error_points], [i[1] for i in error_points], s=10, c="red")
plt.title(f"Error greater than {error_threshold}")
plt.draw()
plt.savefig(f"testoutput/fault3-weakpoints-error{error_threshold}.jpg")