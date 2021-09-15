import os
import pickle
import argparse
from utils import *
from glob import glob
from PIL import Image

def feature_extract(dataroot, result_dir, device):
    feature = []
    path = []

    preprocess = transform()
    model = extractor().to(device)

    #Extract features
    for cls in glob(dataroot+"/*"):
        for exten in ("*.jpg", "*.jpeg", "*.png"):
            for image_path in glob(cls + "/" + exten):
                image = Image.open(image_path)
                tensor = preprocess(image).unsqueeze(0).to(device)
                img_tensor = model(tensor).detach().cpu().squeeze(0)
                feature.append(img_tensor)
                path.append(image_path)

    #Create .pkl file for path 
    with open(f"{result_dir}/path.pickle", "wb") as f:
        pickle.dump(path, f)
    
    #Create .pkl file for feature extraction 
    with open(f"{result_dir}/feature.pickle", "wb") as f:
        pickle.dump(feature, f)

def option():
    parser = argparse.ArgumentParser("Data features extraction")
    parser.add_argument("--dataroot", type=str, required=True, metavar="IN",
                        help="Dataroot for extractor.")
    parser.add_argument("--result_dir", default="./result", type=str, metavar="OUT",
                        help="Output directory for extractor.")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Set gpu mode: [cpu, cuda].")

    return parser.parse_args()

if  __name__ == "__main__":
    print("Querring...")
    parser = option()
    os.makedirs(parser.result_dir)
    feature_extract(*vars(parser).values())
    print("Done!")