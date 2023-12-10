import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from models.models import Brainmodels
from data.data_loader import *
from rsa import *

# Define model names and their corresponding weight file names
models_info = {
    "resnet101": "resnet101.pth",
}

# Define the depth levels for feature extraction
depths = ["f1", "l1", "l2", "l3", "l4"]

# Function to extract a subnetwork based on the model name and feature type
def extended_extract_subnetwork(cnn, model_name, feature_type):
    architectures = {}
    if "resnet" in model_name or "resnext" in model_name:
        architectures[model_name] = {
            "f1": list(cnn.children())[:4],
            "l1": list(cnn.children())[:5],
            "l2": list(cnn.children())[:6],
            "l3": list(cnn.children())[:7],
            "l4": list(cnn.children())[:8],
        }
    else:
        raise ValueError(f"Unknown model name {model_name}.")

    if model_name in architectures:
        if feature_type in architectures[model_name]:
            layers = architectures[model_name][feature_type]
            cn = nn.Sequential(*layers)
            cn = cn.eval().cuda() if torch.cuda.is_available() else cn.eval()
            return cn
        else:
            raise ValueError(f"Unknown feature type {feature_type} for model {model_name}.")
    else:
        raise ValueError(f"Unknown model name {model_name}.")

# Function to test pretrained models and save the generated heatmaps
def pretrained(config):
    test_img = sorted(os.listdir(config.test_ima_path))
    img_inte = sorted(os.listdir(config.test_path), key=lambda l: int(re.findall('\d+', l)[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Now the testing models is {config.m_name}')

    model_folder = os.path.join("./rdm", config.m_name)
    os.makedirs(model_folder, exist_ok=True)

    # Create a directory for .npy files
    npy_folder = os.path.join(model_folder, "npy")
    os.makedirs(npy_folder, exist_ok=True)

    net = Brainmodels("resnet101", config.classes, config.pretrained)
    cnn = net.final_models().to(device)

    for depth in depths:
        edu_distance = []
        test_model = resnet101_cifar(cnn, depth)
        for num in range(len(test_img)):
            x = []
            for num_y in range(len(img_inte)):
                imgpath = os.path.join(config.test_ima_path, test_img[num])
                imgpath1 = os.path.join(config.test_path, img_inte[num_y])
                tmp1 = extract_feature(test_model, imgpath)
                tmp2 = extract_feature(test_model, imgpath1)
                x.append(l2(tmp1, tmp2))
            edu_distance.append(x)

        edu = np.array(edu_distance)
        normalized_edu = MinMaxScaler().fit_transform(edu)
        np.save(os.path.join(npy_folder, f"{depth}.npy"), normalized_edu)

        # Generate and save heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        df = pd.DataFrame(normalized_edu, columns=[str(i) for i in range(1, 9)], index=[str(i) for i in range(1, 9)])
        sns.heatmap(df, annot=False, vmax=1, vmin=0, square=True, cmap="YlGnBu", cbar=False)
        plt.title('Representational Dissimilarity Matrix', fontsize=25)
        plt.ylabel('Subject Perceived Angle (1-8)', fontsize=25)
        plt.xlabel('Visual Illusion Strength (1-8)', fontsize=25)
        plt.xticks(fontsize=31)
        plt.yticks(fontsize=31)

        cbar = ax.figure.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Colorbar Label', rotation=-90, va="bottom", fontsize=21)
        cbar.ax.tick_params(labelsize=21)

        plt.savefig(os.path.join(model_folder, f'{depth}.tif'), format='tif', dpi=300)

# Main function to initialize and run the testing process
if __name__ == "__main__":
    pretrain_info = argparse.Namespace(
        m_name="resnet101",
        checkpoint_path="normal",
        test_path="data/test_a/",
        pretrained=True,
        classes=2,
        test_ima_path="data/test_b/"
    )
    pretrained(pretrain_info)
