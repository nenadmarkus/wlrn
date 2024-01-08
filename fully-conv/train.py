import torch
import time
import os
import math

DEVICE = "cuda"

#
# parse command line options
#

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
parser.add_argument('--writepath', type=str, default=None, help='where to write the learned weights')
parser.add_argument('--learnrate', type=float, default=1e-5, help='RMSprop learning rate')
parser.add_argument('--descdim', type=int, default=256, help='descriptor size')

args = parser.parse_args()

#
# set up the model (image+keypoints -> descriptors)
#

from convdesc import DescriptorExtractor

MODEL = DescriptorExtractor({'weights': args.loadpath, 'descriptor_dim': args.descdim})
MODEL.to(DEVICE)

#
# this is the loss function
#

def compute_skar_loss(AP, AN, thr=0.8, temp=0.025):
    # this is a parameter of the loss
    beta = -math.log(1.0/0.99 - 1)/(1.0-thr)
    # compute similarities and rescale them to [0, 1]
    AP = AP.add(1).mul(0.5)
    AN = AN.add(1).mul(0.5)
    # kill all scores below `thr`
    AP = torch.sigmoid(AP.add(-thr).mul(beta)) #* torch.softmax(AP/temp, dim=0) * torch.softmax(AP/temp, dim=1)
    AN = torch.sigmoid(AN.add(-thr).mul(beta))
    # compute the loss
    M = (1 + torch.sum(torch.max(AN, 1)[0]))/(1 + torch.sum(torch.max(AP, 1)[0]))
    return M

#
# data-loading and handling
#

def prepare_input(inp, device):
    if type(inp) == list:
        return [{
            "image": i["image"].to(device),
            "keypoints": torch.tensor(i["keypoints"]).to(device),
        } for i in inp if i is not None]
    else:
        return {
            "image": inp["image"].to(device),
            "keypoints": torch.tensor(inp["keypoints"]).to(device),
        }

import loader
LOADER = loader.init({
    "data_path": "datasets/ukbench/",
    "use_triplets": True,
    "use_augmentations": False,
    "num_workers": 8,
    "keypoints": "sift",
    "batch_size": 4
})

#
# optimizer and training step
#

OPTIMIZER = torch.optim.RMSprop(MODEL.parameters(), lr=args.learnrate)

def train_step(batch):
    avgloss = 0
    OPTIMIZER.zero_grad()

    for triplet in batch:
        if triplet is None: continue

        # triplet is (anchor, positive, negative) and each of these elements has the following keys: 'keypoints', 'image'
        triplet = [prepare_input(ik, DEVICE) for ik in triplet]

        A = MODEL(triplet[0])
        P = MODEL(triplet[1])
        if type(triplet[2]) == list:
            if len(triplet[2]) == 0:
                continue
            N = torch.cat([MODEL(n) for n in triplet[2] if n is not None])
        else:
            N = MODEL(triplet[2])

        AP = torch.mm( A, P.t() )
        AN = torch.mm( A, N.t() )

        loss = compute_skar_loss(AP, AN)
        loss.backward()

        avgloss = avgloss + loss.item()

    OPTIMIZER.step()
    avgloss = avgloss/len(batch)

    return avgloss

#
# training loop
#

STEP = 0
while True:
    for batch in LOADER:
        if STEP > 500000: quit()
        start = time.time()
        avgloss = train_step(batch)
        STEP = STEP + 1
        print('* step %d finished in %d [ms] (average loss: %f)' % (STEP, int(1000*(time.time()-start)), avgloss))
        if args.writepath and STEP!=0 and 0==STEP%5000:
            os.system('mkdir -p ' + args.writepath)
            path = args.writepath + '/' + str(STEP) + '.pth'
            print('* saving model weights to ' + path)
            torch.save(MODEL.state_dict(), path)