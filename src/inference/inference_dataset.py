import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from src.AIM22_ReverseISP_MiAlgo.model import LiteISPNet_s as NET #Inverse ISP
from src.AIM22_ReverseISP_MiAlgo.utils import demosaic_resize


print("===> Loading network ISP parameters")
device = torch.device("cuda")
model_isp = NET()
checkpoint = torch.load("src/AIM22_ReverseISP_MiAlgo/ckpts/p20.pth")
model_isp.load_state_dict(checkpoint['state_dict_model'])
print("number of gpus: ", torch.cuda.device_count())
model_isp = model_isp.to(device)
model_isp = nn.DataParallel(model_isp)
model_isp = model_isp.eval()


def inverse_ISP(img):        
    H, W = 2976, 2976 #ISP model works better with this input dimension
    img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
    with torch.no_grad():
        output = model_isp(img)
    out = output.detach().cpu().numpy().transpose((0, 2, 3, 1))
    wlevl, blevl = 1020, 60

    out = out * (wlevl - blevl) + blevl
    out = out.round()
    out = np.clip(out, 60, wlevl)
    out = out.astype(np.uint16)    
    output_tensors = []
    # Iterate over each sample in the batch
    for i in range(out.shape[0]):
        sample = out[i]
        resized_sample = demosaic_resize(sample)
        resized_sample_tensor = torch.from_numpy(resized_sample)
        output_tensors.append(resized_sample_tensor)
    out = torch.stack(output_tensors, dim=0)
    out = out.permute(0, 3, 1, 2)
    out = out.cuda()  
    out = F.interpolate(out, size=(380, 380), mode='bilinear', align_corners=False)
    return out


def save_images_as_png(images_tensor, directory='./sanitary_images_ISP'):
    # Sanitary check for input images before feeding to network
    os.makedirs(directory, exist_ok=True)
    for i in range(images_tensor.size(0)):
        image_tensor = images_tensor[i]
        
        # Normalize image tensor to range [0, 1]
        image_tensor = image_tensor - image_tensor.min()
        image_tensor = image_tensor / image_tensor.max()
        image_np = image_tensor.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        plt.imsave(os.path.join(directory, f"image_{i}.png"), image_np)


def main(args):
    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model = nn.DataParallel(model)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
        print(len(video_list))
        #for cross manipulation dataset
    elif args.dataset == 'FF_Deepfakes':
        video_list,target_list=init_ff(dataset='Deepfakes')
        print(len(video_list))
    elif args.dataset == 'FF_Face2Face':
        video_list,target_list=init_ff(dataset='Face2Face')
        print(len(video_list))
    elif args.dataset == 'FF_FaceSwap':
        video_list,target_list=init_ff(dataset='FaceSwap')
        print(len(video_list))
    
    elif args.dataset == 'FF_NeuralTextures':
        video_list,target_list=init_ff(dataset='NeuralTextures')
        print(len(video_list))
        
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError
    output_list=[]
    for filename in tqdm(video_list):
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)
            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                img = inverse_ISP(img)
                img = img.float()                
                pred=model(img).softmax(1)[:,1]
            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)        
    
    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')
    
    
    with open(f'{args.dataset}.txt', 'w') as f:
        f.write(f'{args.dataset}| AUC: {auc:.4f}\n')


if __name__=='__main__':
    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames', default=10,type=int)
    args=parser.parse_args()
    print("number of frames/video: ", args.n_frames)
    main(args)

