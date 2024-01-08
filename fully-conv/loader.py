import os
import numpy
import cv2
import random
import torch

cv2.setNumThreads(0)

STATE = {
    "superpoint": None
}

#
#
#

def kpts_to_xy(kpts):
    return [[k.pt[0], k.pt[1]] for k in kpts]

def draw_xy(image, xy_arr):
    for xy in xy_arr:
        cv2.circle(image, (int(xy[0]), int(xy[1])), 2, color=(0, 255, 0), thickness=-1)

def get_orb_keypoints(image, maxn, draw=False):
    orb = cv2.ORB_create(nfeatures=maxn, nlevels=3)
    kpts = orb.detect(image, None)
    if draw: cv2.drawKeypoints(image, kpts, image, color=(0, 0, 255))
    return kpts_to_xy(kpts)

def get_sift_keypoints(image, maxn, draw=False):
    sift = cv2.SIFT_create(nfeatures=maxn)
    kpts = sift.detect(image, None)
    if draw: cv2.drawKeypoints(image, kpts, image, color=(0, 0, 255))
    return kpts_to_xy(kpts)

def get_superpoint_keypoints(image, maxn, draw=False, device="cuda:1"):
    cuda = torch.cuda.is_available()
    if STATE["superpoint"] is None:
        if not os.path.exists("superpoint.py"):
            os.system("curl https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superpoint.py --output superpoint.py")
        if not os.path.exists("weights/superpoint_v1.pth"):
            os.system("mkdir -p weights/ && curl https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superpoint_v1.pth --output weights/superpoint_v1.pth")
        from superpoint import SuperPoint
        STATE["superpoint"] = SuperPoint({"max_keypoints": maxn, "weights": "superpoint_v1.pth"})
        STATE["superpoint"].eval()
        if cuda: STATE["superpoint"].to(device)
    i = torch.from_numpy(image).float().div(255.0)[:, :, 1] # take just the green channel
    if cuda: i = i.to(device)
    with torch.no_grad():
        xy = STATE["superpoint"].forward({"image": i.unsqueeze(0).unsqueeze(0)})["keypoints"][0].cpu().tolist()
    if draw: draw_xy(image, xy)
    return xy

#
#
#

def apply_augmentation(image):
    shape = image.shape[0:2]
    angle = 2.0*(numpy.random.random() - 0.5) * 180 / 6 # rotate by 30 deg max
    scale = 0.9 + 0.6*numpy.random.random()
    A = cv2.getRotationMatrix2D([c/2.0 for c in image.shape[0:2][::-1]], angle, scale)
    p1 = 0.001*(numpy.random.random()-0.5)
    p2 = 0.001*(numpy.random.random()-0.5)
    H = numpy.array([
        [A[0, 0], A[0, 1], A[0, 2]],
        [A[1, 0], A[1, 1], A[1, 2]],
        [p1, p2, 1.0]
    ])
    warp = cv2.warpPerspective(image, H, shape[::-1])
    return warp

def load_paths_and_labels_ukbench(folder):
    p_and_l = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(".jpg"):
                p = os.path.join(root, filename)
                l = int(filename.split(".jpg")[0].split("ukbench")[1]) // 4
                p_and_l.append((p, l))
    return p_and_l

# define the function
def get_sample(D, triplets=True, numnegs=16):
    keys = list(D.keys())
    random.shuffle(keys)
    # matching pair
    vals = D[keys[0]]
    random.shuffle(vals)
    rez = vals[0:2]
    # non-matching sample
    if triplets:
        if numnegs > 1:
            negs = []
            for i in range(1, 1 + numnegs):
                negs.append(random.choice(D[keys[i]]))
            rez.append(negs)
        else:
            rez.append(random.choice(D[keys[1]]))
    # we're done
    return rez

def init(config):
    if "ukbench" in config["data_path"]:
        p_and_l = load_paths_and_labels_ukbench(config["data_path"])
    else:
        raise Exception("Invalid dataset")

    if "keypoints" not in config:
        config["keypoints"] = "superpoint"

    if "max_keypoints" not in config:
        config["max_keypoints"] = 4096

    print("loader: ", config)

    def load_from_impath(path):
        # prepare the image
        image = cv2.imread(path)
        if config["use_augmentations"]:
            image = apply_augmentation(image)
        # prepare keypoints
        if config["keypoints"] == "superpoint":
            kp = get_superpoint_keypoints(image, config["max_keypoints"], draw=False)
        if config["keypoints"] == "sift":
            kp = get_sift_keypoints(image, config["max_keypoints"], draw=False)
        if config["keypoints"] == "orb":
            kp = get_orb_keypoints(image, config["max_keypoints"], draw=False)
        if len(kp) < 16:
            print("* low keypoint count for '%s'" % path)
            return None
        # we're done
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float().div(255),
            "keypoints": kp
        }

    # make the sampling dict
    D = {}
    for (p, l) in p_and_l:
        print(p, l)
        if l not in D:
            D[l] = []
        D[l].append(p)

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, D, use_triplets):
            self.D = D
            self.use_triplets = use_triplets
        def __len__(self):
            return len(self.D)*10
        def __getitem__(self, index):
            imps = get_sample(self.D, self.use_triplets)

            rez = [
                load_from_impath(imps[0]),
                load_from_impath(imps[1]),
                load_from_impath(imps[2]) if type(imps[2]) is not list else [load_from_impath(p) for p in imps[2]]
            ]

            if any([s is None for s in rez]):
                return None
            else:
                return rez

    def collate_fn(batch):
        return batch

    return torch.utils.data.DataLoader(MyDataset(D, config["use_triplets"]), batch_size=config["batch_size"], num_workers=config["num_workers"], collate_fn=collate_fn, shuffle=True)

#
#
#

def test_1():
    data = load_paths_and_labels_ukbench("ukbench")
    for p, l in data:
        i = cv2.imread(p)
        print(l)
        cv2.imshow("img", i)
        if ord('q') == cv2.waitKey(0): break

def test_2():
    data = load_paths_and_labels_ukbench("ukbench")
    for p, l in data:
        i = apply_augmentation(cv2.imread(p))
        print(l)
        cv2.imshow("img", i)
        if ord('q') == cv2.waitKey(0): break

def test_3():
    data = load_paths_and_labels_ukbench("ukbench")
    for p, l in data:
        i = apply_augmentation(cv2.imread(p))
        #k = get_orb_keypoints(i, 256)
        k = get_sift_keypoints(i, 256)
        print(l)
        draw_xy(i, k)
        cv2.imshow("img", i)
        if ord('q') == cv2.waitKey(0): break

def test_4():
    loader = init({
        "data_path": "datasets/ukbench/",
        "use_augmentations": True,
        "use_triplets": True,
        "num_workers": 8,
        "batch_size": 4
    })

    for batch in loader:
        i = batch[0][0]["image"]
        print(i.shape)
        break

if __name__ == "__main__":
    test_4()
