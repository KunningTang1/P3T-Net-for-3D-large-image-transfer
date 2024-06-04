import argparse
from skimage import io
import torch
from models import ResnetGenerator, ResnetGenerator1
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
import numpy as np
from Semantic_Indication_Modules import uresnet,uresnet1
import os
from tqdm import tqdm
from collections import OrderedDict
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)            
		
T1 = [440,460,480,500,520,539,560,580,600,620,640,660,680,700,720,740,760,780,800,814]  
# T1 = [539] 
for T in T1:
    print(T)
    file = "H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\" + str(T)
    mkdir(file)    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    opt = parser.parse_args()
    
    im_tfs = tfs.Compose([
          tfs.ToTensor(),
          ])
    # Domain tranfer at X-Y plane
    netG_B2A =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks=4)
    checkpoint = torch.load('H:\\DomainTransfer\\Case4\\checkpoints\\ck_CycSem\\39.pt')
    netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])
    netG_B2A.cuda()
    noise = io.imread('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\scan126' + str(T)+'_8bits_registered.tif')
    noise = noise[:,4:764,5:765]
    io.imsave('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\'+ str(T)+ '\\'+ str(T)+'raw.tif',noise)
    
    output = np.zeros((np.size(noise,0),np.size(noise,1),np.size(noise,2)),dtype = np.uint8)
    for i in tqdm(range(np.size(noise,0))):
        im = noise[i,:,:]
        cut_image1 = im_tfs(im)
        test_image1 = cut_image1.unsqueeze(0).float()
        with torch.no_grad():
            out = netG_B2A(test_image1.cuda())
        out  = out.clamp(min=0, max=1)
        pred = np.uint8(out.squeeze().data.cpu().numpy()*255)
        output[i,:,:] = pred
    io.imsave('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\'+ str(T)+ '\\'+ str(T)+'CS.tif',output)
    
    #%% Thrid axis fixing
    netG_B_stacking =  ResnetGenerator1(opt.output_nc, opt.input_nc,n_blocks=4)
    checkpoint = torch.load('H:\\DomainTransfer\\Case4\\checkpoints\\ck_CSFNet_TT\\55.pt')
    netG_B_stacking.load_state_dict(checkpoint['netG_A_stacking_state_dict'])
    netG_B_stacking.cuda()
    ###################################
    vol = io.imread('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\'+ str(T)+ '\\'+ str(T)+'CS.tif')
    after = np.zeros([np.size(vol,0),np.size(vol,1),np.size(vol,2)])
    for i in tqdm(range(0,np.size(vol,1))):
        img = vol[:,i,:]
        B = im_tfs(img)
        B =B.unsqueeze(1)
        with torch.no_grad():
            out = netG_B_stacking(B.cuda())
        out = out.clamp(min=0, max=1)
        fake_A_xy1 = np.uint8(np.squeeze(out).cpu().detach().numpy()*255.0)
        after[:,i,:] = fake_A_xy1
    io.imsave('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\'+ str(T)+ '\\'+ str(T)+'CSF.tif',np.uint8(after))
    #%% Semantic segmentation using sementic indication module
    net = uresnet()
    checkpoint = torch.load('H:\\DomainTransfer\\Case4\\checkpoints\\ck_pretrainSegnet_TT\\13.pt')
    net.load_state_dict(checkpoint['net_state_dict'])
    net1 = uresnet1()
    net1_dict = net1.state_dict()
    new_state_dict = OrderedDict()
    new_state_dict = {k:v for k,v in checkpoint['net_state_dict'].items() if k in net1_dict}
    net1_dict.update(new_state_dict)
    net1.load_state_dict(net1_dict)
    net1.cuda()
    
    net1.eval()
    img = io.imread('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\'+ str(T)+ '\\'+ str(T)+'CSF.tif')
    seg = np.zeros((np.size(img,0),np.size(img,1),np.size(img,2)))
    net1.eval()
    for i in tqdm(range(0,np.size(img,0))):
        img1 = img[i,:,:]
        cut_image1 = im_tfs(img1)
        test_image1 = cut_image1.unsqueeze(0).float()
        with torch.no_grad():
            out = net1(test_image1.cuda())
        pred = out.max(1)[1].squeeze().data.cpu().numpy()
        pred1 = np.uint8(pred)
        seg[i,:,:] = pred1
    io.imsave('H:\\DomainTransfer\\Case4\\raw3Dimage\\Drainage\\OneDrive_1_7-19-2023\\Results\\'+ str(T)+ '\\'+ str(T)+'CSF_SEG.tif',np.uint8(seg))
    

    
#%% 

