import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from utils.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector
from torch.utils.tensorboard import SummaryWriter
from src.AIM22_ReverseISP_MiAlgo.model import LiteISPNet_s as NET #Inverse ISP
from src.AIM22_ReverseISP_MiAlgo.utils import demosaic_resize
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

print("===> Loading network ISP parameters")
device = torch.device("cuda")
model_isp = NET()
checkpoint = torch.load("src/AIM22_ReverseISP_MiAlgo/ckpts/p20.pth")
model_isp.load_state_dict(checkpoint['state_dict_model'])
model_isp = model_isp.to(device)
model_isp = nn.DataParallel(model_isp)
model_isp = model_isp.eval()



def inverse_ISP(img):
    #input image to inverse isp        
    H, W = 2976, 2976 #ISP model works better with this input dimension
    img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
    with torch.no_grad():
        output = model_isp(img) #out -1:  torch.Size([8, 4, 1488, 1488])
    out = output.detach().cpu().numpy().transpose((0, 2, 3, 1)) #[0]
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
    out = out.cuda()  # Assuming you're using GPU
    out = F.interpolate(out, size=(380, 380), mode='bilinear', align_corners=False)
    return out


def save_images_as_png(batch, images_tensor, directory='./sanitary_images_ISP'):
    # Sanitary check for input images feeding to network
    os.makedirs(directory, exist_ok=True)
    # Loop through each image in the tensor
    for i in range(images_tensor.size(0)):
        # Extract the image from the tensor
        image_tensor = images_tensor[i]
        
        # Normalize image tensor to range [0, 1]
        image_tensor = image_tensor - image_tensor.min()
        image_tensor = image_tensor / image_tensor.max()
        image_np = image_tensor.cpu().numpy()
       
        
        # Transpose the array to match Height x Width x Channels
        image_np = np.transpose(image_np, (1, 2, 0))
        
        # Save the NumPy array as a PNG file
        plt.imsave(os.path.join(directory, f"{batch}_image_{i}.png"), image_np)


def save_mask(batch, images_tensor, directory='./sanitary_mask'):
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    for i in range(images_tensor.size(0)):
        image_tensor = images_tensor[i]
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        image_np = image_tensor.float().cpu().numpy()
        plt.imsave(os.path.join(directory, f"{batch}_mask_{i}.png"), image_np, cmap='gray')


def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    cfg=load_json(args.config)

    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')


    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    train_dataset=SBI_Dataset(phase='train',image_size=image_size)
    val_dataset=SBI_Dataset(phase='val',image_size=image_size)
   
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=16, #sahar from 4 to 16
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=16,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    
    model=Detector()
    model=model.to('cuda')

    if args.pretrained_model:
        cnn_sd=torch.load(args.weight_name)["model"]
        model.load_state_dict(cnn_sd)
        print(args.weight_name)
        
    model = nn.DataParallel(model) 


    iter_loss=[]
    train_losses=[]
    test_losses=[]
    train_accs=[]
    test_accs=[]
    val_accs=[]
    val_losses=[]
    n_epoch=cfg['epoch']
    lr_scheduler=LinearDecayLR(model.module.optimizer, n_epoch, int(n_epoch/4*3)) 

    last_loss=99999


    now=datetime.now()
    save_path='output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion=nn.CrossEntropyLoss()

    writer = SummaryWriter() #writes in runs by default 
    last_auc=0
    last_val_auc=0
    weight_dict={}
    n_weight=5
        

    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train(mode=True)
        for step, data in enumerate(tqdm(train_loader)):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            mask=data['mask'] #get mask
            #================================
            #Sanitary check for input images rgb
            # save_images_as_png(step, img, directory='./sanitary_images_RGB')
            #================================
            img = inverse_ISP(img)
            img = img.float()
            # output=model.training_step(img, target)#sahar
            output=model.module.training_step(img, target)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            train_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
            #================================
            #Sanitary check for input images raw
            # save_images_as_png(step, img)
            # save_mask(step, mask)
            # if step == 100:
            #     quit()
            #================================        
  
        lr_scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader))

        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
                        epoch+1,
                        n_epoch,
                        train_loss/len(train_loader),
                        train_acc/len(train_loader),
                        )
        # Log training loss and accuracy for each epoch
        writer.add_scalar('Train Loss/epoch', train_loss/len(train_loader), epoch)
        writer.add_scalar('Train Accuracy/epoch', train_acc/len(train_loader), epoch)
        # grid = torchvision.utils.make_grid(img)
        # writer.add_image('images', grid, epoch)
        # writer.add_graph(model, img)
        
        model.train(mode=False)
        val_loss=0.
        val_acc=0.
        output_dict=[]
        target_dict=[]
        np.random.seed(seed)
        
        for step,data in enumerate(tqdm(val_loader)):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            
            with torch.no_grad():
                output=model(img)
                loss=criterion(output,target)
            
            loss_value=loss.item()
            iter_loss.append(loss_value)
            val_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            val_acc+=acc
            output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
            target_dict+=target.cpu().data.numpy().tolist()
                    
        val_losses.append(val_loss/len(val_loader))
        val_accs.append(val_acc/len(val_loader))
        val_auc=roc_auc_score(target_dict,output_dict)
        log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
                        val_loss/len(val_loader),
                        val_acc/len(val_loader),
                        val_auc
                        )

        writer.add_scalar('Validation Loss/epoch', val_loss / len(val_loader), epoch)
        writer.add_scalar('Validation Accuracy/epoch', val_acc / len(val_loader), epoch)
        writer.add_scalar('Validation AUC epoch', val_auc, epoch)

        if len(weight_dict)<n_weight:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            weight_dict[save_model_path]=val_auc
            torch.save({
                    "model":model.state_dict(),
                    "optimizer":model.module.optimizer.state_dict(),
                    "epoch":epoch
                },save_model_path)
            last_val_auc=min([weight_dict[k] for k in weight_dict])

        elif val_auc>=last_val_auc:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            for k in weight_dict:
                if weight_dict[k]==last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path]=val_auc
                    break
            torch.save({
                    "model":model.state_dict(),
                    "optimizer":model.module.optimizer.state_dict(),
                    "epoch":epoch
                },save_model_path)
            last_val_auc=min([weight_dict[k] for k in weight_dict])
        
        logger.info(log_text)
        
        
    writer.close()
if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name')
    parser.add_argument('--pretrained_model', action='store_true', help='Load the pretrained checkpoint if True')
    parser.add_argument('-w',dest='weight_name',type=str) 

    args=parser.parse_args()
    main(args)
        
