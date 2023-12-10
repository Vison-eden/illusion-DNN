import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from numpy.linalg import norm
from PIL import Image
import os
import torch.nn.functional as F
from scipy import spatial


img_to_tensor = transforms.ToTensor()
ch_norm_mean = (0.5, 0.5, 0.5)
ch_norm_std = (0.5, 0.5, 0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=ch_norm_mean, std=ch_norm_std)
])

def ave_list(al, le):
    ave = []
    for para in range(len(al)):
        ave.append(al[para]/le)
    return ave



def real_list(a, b):
    a[:] = [a[i] if i < len(a) else 0 for i, j in enumerate(b)]
    result = [sum(n) for n in zip(a, b)]
    return result



def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def final_list(first_list, num):
    temp = func(first_list, num)
    last_list = []
    for li in temp:
        last_list.append(np.mean(li))
    return last_list


def final_ave(avelist, num):
    temp = func(avelist, num)
    list_ave = []
    for li in temp:
        fi_list = []
        len_li = None
        for le in range(len(li)):
            len_li = len(li)
            fi_list = real_list(fi_list, li[le])
        list_ave.append(ave_list(fi_list, len_li))
    return list_ave


# Function to extract features from an image using a model
def extract_feature(model, imgpath):
    img = Image.open(imgpath)  # Load the image
    img = img.resize((560,160))
    tensor = transform(img)
    tensor = tensor.cuda()  # Remove this line if running on CPU
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    result = model(Variable(tensor))
    result = result.data.cpu().numpy()
    return result[0]

# Function to calculate cosine similarity between two vectors
def cos(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    v1 = v1 / 255  # Normalizing the vectors
    v2 = v2 / 255
    similarity = -1 * (spatial.distance.cosine(v1, v2) - 1)
    return similarity

# Function to calculate L2 distance between two vectors
def l2(v1, v2):
    return np.linalg.norm(v1 - v2)

# Function to calculate L1 distance between two vectors
def l1(v1, v2):
    return np.linalg.norm(v1 - v2, ord=1)

# Function to calculate cosine similarity between two vectors
def vector_vector(arr, brr):
    return np.sum(arr * brr) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr)))

# Function to calculate cosine similarity between a vector and a matrix
def vector_matrix(arr, brr):
    return arr.dot(brr.T) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr, axis=1)))

# Function to calculate cosine similarity between two matrices
def matrix_matrix(arr, brr):
    return np.sum(arr * brr, axis=1) / (np.sqrt(np.sum(arr ** 2, axis=1)) * np.sqrt(np.sum(brr ** 2, axis=1)))

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(file))
    return L
