import argparse
from skimage import io
import torch
from models import ResnetGenerator
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
import numpy as np
from models import ResnetGenerator, UResNet,Uresnet
import os
from tqdm import tqdm
from collections import OrderedDict
from os import listdir
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)        
        
directories = [
    "../Example/Inference",]
for directory in directories:
    mkdir(directory)		
#%%
#SCDT: Semantic-consistent domain transfer module
parser = argparse.ArgumentParser()
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
parser.add_argument('--dir_checkpoint_SCDT', type=str, default='../Example/Domain Transfer Module/checkpoints',
                    help='Loading checkpoints directory SCDT')
parser.add_argument('--ck_SCDT', type=int, default=150, help='load the index of checkpoints')
parser.add_argument('--dir_checkpoint_MF', type=str, default='../Example/Misalignment fixing module/checkpoints',
                    help='Loading checkpoints directory for MF')
parser.add_argument('--ck_MF', type=int, default=8, help='load the index of checkpoints')
parser.add_argument('--dir_checkpoint_SI', type=str, default='../Example/Semantic Indication Module/checkpoints',
                    help='Loading checkpoints directory for SI')
parser.add_argument('--ck_SI', type=int, default=40, help='load the index of checkpoints')
parser.add_argument('--gpu', type=int, default=1, help='define which gpu yo use, can be "0" or "1" if you have two gpus')
opt = parser.parse_args()

im_tfs = tfs.Compose([
      tfs.ToTensor(),
      ])
#%% Step1: Semantic-consistent Domain tranfer at X-Y plane
netG_B2A =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks=4)
checkpoint = torch.load(os.path.join(opt.dir_checkpoint_SCDT, '{}.pt'.format(opt.ck_SCDT)),weights_only=True)
netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])
netG_B2A.cuda(opt.gpu)

path_data = '../Testingdata'
num_volumes = listdir(path_data)
sd = io.imread(os.path.join(path_data, num_volumes[0]))
output = np.zeros((np.size(sd,0),np.size(sd,1),np.size(sd,2)),dtype = np.uint8)
for i in tqdm(range(np.size(sd,0))):
    im = sd[i,:,:]
    cut_image1 = im_tfs(im)
    test_image1 = cut_image1.unsqueeze(0).float()
    with torch.no_grad():
        out = netG_B2A(test_image1.cuda(opt.gpu))
    out  = out.clamp(min=0, max=1)
    pred = np.uint8(out.squeeze().data.cpu().numpy()*255)
    output[i,:,:] = pred
io.imsave(directories[0] +'/Transferred_SourceDomain.tif', output)
#%% Step1: Misalignment fixing module
netG_A_stacking =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks=4)
checkpoint = torch.load(os.path.join(opt.dir_checkpoint_MF, '{}.pt'.format(opt.ck_MF)),weights_only=True)
netG_A_stacking.load_state_dict(checkpoint['netG_A_stacking_state_dict'])
netG_A_stacking.cuda(opt.gpu)

###################################
vol = io.imread(directories[0] +'/Transferred_SourceDomain.tif')
after = np.zeros([np.size(vol,0),np.size(vol,1),np.size(vol,2)])
for i in tqdm(range(0,np.size(vol,1))):
    img = vol[:,i,:]
    B = im_tfs(img)
    B =B.unsqueeze(1)
    with torch.no_grad():
        out = netG_A_stacking(B.cuda(opt.gpu))
    out = out.clamp(min=0, max=1)
    fake_A_xy1 = np.uint8(np.squeeze(out).cpu().detach().numpy()*255.0)
    after[:,i,:] = fake_A_xy1
io.imsave(directories[0] +'/Transferred_SourceDomain_MF.tif',np.uint8(after))
#%% Step 3: Semantic segmentation using sementic indication module

net = Uresnet(opt.input_nc,opt.num_classes)  
net1 = UResNet(opt.input_nc,opt.num_classes)
checkpoint = torch.load(os.path.join(opt.dir_checkpoint_SI, '{}.pt'.format(opt.ck_SI)),weights_only=True)
net.load_state_dict(checkpoint['net_state_dict'])
net1_dict = net1.state_dict()
new_state_dict = OrderedDict()
new_state_dict = {k:v for k,v in checkpoint['net_state_dict'].items() if k in net1_dict}
net1_dict.update(new_state_dict)
net1.load_state_dict(net1_dict)
net1.cuda(opt.gpu)

net1.eval()

img = io.imread(directories[0] +'/Transferred_SourceDomain_MF.tif')
img  = img[:,4:,4:]
seg = np.zeros((np.size(img,0),np.size(img,1),np.size(img,2)))
for i in tqdm(range(0,np.size(img,0))):
    img1 = img[i,:,:]
    cut_image1 = im_tfs(img1)
    test_image1 = cut_image1.unsqueeze(0).float()
    with torch.no_grad():
        out = net1(test_image1.cuda(opt.gpu))
    pred = out.max(1)[1].squeeze().data.cpu().numpy()
    pred1 = np.uint8(pred)
    seg[i,:,:] = pred1
io.imsave(directories[0] +'/Transferred_SourceDomain_MF_SEG.tif',np.uint8(seg))
#%% 

