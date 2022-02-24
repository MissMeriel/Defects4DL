import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
from models import DAVE2pytorch
from DatasetGenerator import DataSequence
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("testsetpath", type=Path)
    return parser.parse_args()

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelpath = "model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-28Ksamples-135x240-noiseflipblur.pt"
model = torch.load(modelpath, map_location=device)
dataset = DataSequence(args.testsetpath, transform=Compose([ToTensor()]))

print("Retrieving output distribution....")
# print("Moments of distribution:", dataset.get_outputs_distribution())
print("Total samples:", dataset.get_total_samples())


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

BATCH_SIZE=1
testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=worker_init_fn)
import re
# testdir = f"{os.getcwd().strip('tests')}error-segments/seg3"
testdir = "seg3-testoutput"
if not os.path.exists(testdir):
    os.mkdir(testdir)
for i, hashmap in enumerate(testloader, 0):
    sample_num = int(hashmap['filename'][0].strip("sample-").strip(".jpg"))
    # sample_num = int(re.search(r'\d+', sample_num).group())
    pos = [float(s) for s in hashmap['position'][0][1:-1].split("  ")]
    dist_to_fail = math.sqrt(math.pow(pos[0]-177.370,2) + math.pow(pos[1] - -111.850,2))
    if dist_to_fail < 5 and sample_num < 1779:
        x = hashmap['image'].float().to(device)
        y = hashmap['steering_input'].float().to(device)
        outputs = model(x)
        print(f"{hashmap['filename'][0]}\tcorrect output={y.item():.3f}\tmodel output={outputs.item():.3f}\t\terror={outputs.item()-y.item():.3f}")
        if abs(outputs.item() - y.item()) > 0.75:
            plt.title(f"error={outputs.item()-y.item():.3f}", color="red")
        else:
            plt.title(f"error={outputs.item() - y.item():.3f}", color="black")
        print(f"{pos=}\n")
        plt.imshow(x[0].permute(1,2,0))
        plt.savefig(f"{testdir}/sample-{sample_num}-error.jpg")
    # what constitutes a static error? left turn when close to the left side of the track?

    # need track datapoints for left/right/center


    # model_name = "model-DAVE2v3-lr1e4-50epoch-batch64-lossMSE-25Ksamples.pt" #orig, NIER results DNN
    # model_name = "model-DAVE2PytorchModel-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    # model_name = "model-DAVE2v2-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"