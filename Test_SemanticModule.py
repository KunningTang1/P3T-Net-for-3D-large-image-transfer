import torch
import torchvision.transforms as tfs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import UResNet,Uresnet
from tqdm import tqdm
import os
import argparse
from skimage import io
from os import listdir
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UResNet', 
                    help='define which gpu yo use, can be "UresNet"')
parser.add_argument('--cpu_gpu', type=str, default='gpu', 
                    help='define which gpu yo use, can be "gpu" or "cpu"')
parser.add_argument('--num_classes', type=int, default=3, help='Size for inference')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
# This size definition is very important, for now, please use integer multiples of the training patch size
parser.add_argument('--size', type=int, default=448, help='Size for inference')
#===============================================================================================
parser.add_argument('--checkpoint', type=int, default=60, help='Load checkpoint')
parser.add_argument('--gpu', type=int, default=1, help='define which gpu yo use')
parser.add_argument('--dir_checkpoint', type=str, default='../Example/Semantic Indication Module/checkpoints',
                    help='Trained model directory')


path_data = '../Testingdata'
num_volumes = listdir(path_data)
td = io.imread(os.path.join(path_data, num_volumes[1]))

opt = parser.parse_args()
print(opt)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)   
        
if opt.model == 'UResNet':
    net = Uresnet(opt.input_nc,opt.num_classes)  
    net1 = UResNet(opt.input_nc,opt.num_classes)
    print('============== UresNet is used ===================')


checkpoint = torch.load(os.path.join(opt.dir_checkpoint, '{}.pt'.format(opt.checkpoint)),weights_only=True)
net.load_state_dict(checkpoint['net_state_dict'])
if opt.cpu_gpu == 'gpu':
    net.cuda()
    print('============== GPU is used ===================')

net1_dict = net1.state_dict()
new_state_dict = OrderedDict()
new_state_dict = {k:v for k,v in checkpoint['net_state_dict'].items() if k in net1_dict}
net1_dict.update(new_state_dict)
net1.load_state_dict(net1_dict)
net1.cuda()
net1.eval()

im_tfs = tfs.Compose([
      tfs.ToTensor(),
      ])    

def crop(data,height=opt.size, width=opt.size):
    st_x = 0
    st_y = 0
    box = (st_x, st_y, st_x+width, st_y+height)
    data = data.crop(box)
    return data

im = td[300,:,:]
img2 = Image.fromarray(im)
img2 = crop(img2)
cut_image = img2

plt.subplot(121)
plt.imshow(cut_image,cmap='gray')

cut_image1 = im_tfs(cut_image)
test_image1 = cut_image1.unsqueeze(0).float()
with torch.no_grad(): 
    out = net1(test_image1.cuda())
pred = out.max(1)[1].squeeze().data.cpu().numpy()
pred = np.uint8(pred)
plt.subplot(122)
plt.imshow(pred)

    

