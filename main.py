import torch
import pickle
import argparse
from utils import *
from PIL import Image
import matplotlib.pyplot as plt


def calc_similarity(dataroot, querry, device, k):
    feature =  open(f"{dataroot}/feature.pickle",'rb')
    feature = pickle.load(feature)

    preprocess = transform()
    model = extractor().to(device)

    #Preprocess querry image
    q = Image.open(querry)
    q = preprocess(q).unsqueeze(0).to(device)
    q = model(q)

    #Stack an Feature to Tensor shape Nx4096(for VGG16)
    feature = torch.stack(feature).to(device)
    
    #Expand from 1x4096 to Nx4096
    q = q.repeat(feature.size(0),1)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    result = cos(q, feature)

    values, indices = torch.sort(result, descending=True)[:k]
    output = [(values[i].item(), indices[i].item()) for i in range(k)]
    return output

def show_image(dataroot, similarity):
    path = open(f"{dataroot}/path.pickle",'rb')
    path = pickle.load(path)

    fig = plt.figure(figsize=(15, 10))    
    columns = 5 
    row = int(len(similarity) / columns + 1) if len(similarity) % columns else int(len(similarity) / columns)
    
    for i, rank in enumerate(similarity):
        plt.subplot(row, columns, i + 1)
        image = plt.imread(path[rank[1]])
        plt.imshow(image, aspect="auto")
        plt.title(f"Top {i+1}: {rank[0]:.4f}",  fontsize=15, family="fantasy")
        plt.axis('off')

    plt.suptitle(f"Top {len(similarity)} images!", fontsize=30, family="fantasy")
    plt.show()

    fig.savefig("sample.jpg", format='jpg', bbox_inches='tight', facecolor='#889EAF')

def option():
    parser = argparse.ArgumentParser("Data features extraction")
    parser.add_argument("--pkl_dir", type=str, required=True, metavar="IN",
                        help="Directory of pkl files.")
    parser.add_argument("--querry",  type=str,  required=True, metavar="Q",
                        help="Output directory for extractor.")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Set gpu mode: [cpu, cuda].")
    parser.add_argument("--k", default=10, type=int,
                        help="Top k results.")                    

    return parser.parse_args()

if  __name__ == "__main__":
    parser = option()
    sim = calc_similarity(*vars(parser).values())
    show_image(parser.pkl_dir, sim)
    print("Done!")
