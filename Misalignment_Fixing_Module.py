import argparse
import sys
from torch.utils.data import DataLoader
import torch
from models import ResnetGenerator
from utils import ReplayBuffer,weights_init_normal2d,LambdaLR
import numpy as np
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from torchvision.transforms import Compose,ToTensor
from skimage import io
import torchvision.transforms as tfs
import skimage
from models import Discriminator
import os
import matplotlib.pyplot as plt
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--gpu', type=int, default=0, 
                    help='define which gpu yo use, can be "0" or "1" if you have two gpus')
parser.add_argument('--size', type=int, default=96, help='size of the data crop (squared assumed)')
parser.add_argument('--dir_train_data', type=str, default='../Example/Misalignment fixing module/Train/',
                    help='dataset directory')

parser.add_argument('--dir_checkpoint', type=str, default='../Example/Misalignment fixing module/checkpoints',
                    help='Saving checkpoints directory')

parser.add_argument('--dir_test_data', type=str, default='../Example/Misalignment fixing module/Test/',
                    help='dataset directory')

parser.add_argument('--dir_checkpoint_DomainTransfer', type=str, default='../Example/Domain Transfer Module/checkpoints',
                    help='Saving checkpoints directory')
parser.add_argument('--ck_DomainTransfer', type=int, default=150, help='load checkpoint for domain transfer module')
opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
#%% Dataloader
    
class TrainDatasetFromFolder(Dataset):
    def __init__(self, root):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames1 = [join(root, x) for x in listdir(root)][0]
        self.image_filenames2 = [join(root, x) for x in listdir(root)][1]

        LIST1 = listdir(self.image_filenames1)
        LIST2 = listdir(self.image_filenames2)
        # LIST3 = listdir(self.image_filenames3)
        LIST1.sort(key=lambda x: int(x[:-4]))
        LIST2.sort(key=lambda x: int(x[:-4]))
        # LIST3.sort(key=lambda x: int(x[:-4]))
        self.img1 = [join(self.image_filenames1, x) for x in LIST1]
        self.img2 = [join(self.image_filenames2, x) for x in LIST2]
        # self.img3 = [join(self.image_filenames3, x) for x in LIST3]
        self.train_input_transform = transform()
        self.train_output_transform = transform()

        
    def __getitem__(self, index):
        input_image = self.train_input_transform(io.imread(self.img1[index]))
        output_image = self.train_output_transform(io.imread(self.img2[index]))
        return input_image,output_image

    def __len__(self):
        return len(self.img1)
def transform():
    return Compose([
        ToTensor()       
    ])

#%% NetWork     

netG_B2A =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks=4)
netG_B2A1 =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks=4)

netG_A_stacking = ResnetGenerator(opt.input_nc, opt.output_nc,n_blocks=4)
netG_A_stacking1 = ResnetGenerator(opt.input_nc, opt.output_nc,n_blocks=4)
netD_A_xz = Discriminator(opt.output_nc)

checkpoint = torch.load(os.path.join(opt.dir_checkpoint_DomainTransfer, '{}.pt'.format(opt.ck_DomainTransfer)),weights_only=True)
netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])
netG_B2A1.load_state_dict(checkpoint['netG_B2A_state_dict'])
for param in netG_B2A.parameters():
    param.requires_grad = False

netG_B2A.cuda(opt.gpu)
netG_B2A1.cuda(opt.gpu)
netD_A_xz.cuda(opt.gpu)
netG_A_stacking.cuda(opt.gpu)

netG_A_stacking.apply(weights_init_normal2d)
netD_A_xz.apply(weights_init_normal2d);

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G_stack = torch.optim.Adam(netG_A_stacking.parameters(),lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A_xz = torch.optim.Adam(netD_A_xz.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G_stack = torch.optim.lr_scheduler.LambdaLR(optimizer_G_stack, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A_xz = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A_xz, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
fake_A_xy_buffer = ReplayBuffer();fake_A_xyz_buffer = ReplayBuffer();fake_A_yz_buffer = ReplayBuffer()
fake_B_xy_buffer = ReplayBuffer();fake_B_xyz_buffer = ReplayBuffer();fake_B_yz_buffer = ReplayBuffer();
fake_A_xyz_xy_buffer = ReplayBuffer()
feature_fakeA_buffer = ReplayBuffer()

# Dataset loader
train_set = TrainDatasetFromFolder(opt.dir_train_data)
train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

###################################
k = 0
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i,  (real_A, real_B) in enumerate(train_loader):
        real_A = real_A.cuda(opt.gpu)
        real_B = real_B.cuda(opt.gpu)

        real_A = real_A.permute(2, 0, 3, 1)
        real_B = real_B.permute(2, 0, 1, 3)
        
        optimizer_G_stack.zero_grad()
        target_real = torch.ones(real_A.size(0), device=device, requires_grad=False).cuda(opt.gpu)
        target_fake = torch.zeros(real_B.size(0), device=device, requires_grad=False).cuda(opt.gpu)
        
        fake_A_xy = netG_B2A(real_B)
        
        # GAN loss
        real_A_xz = torch.transpose(real_A, 0, 2)
        same_A = netG_A_stacking(real_A_xz)
        
        same_A_xy = torch.transpose(same_A, 0, 2);
        loss_identity_A = criterion_identity(same_A_xy, real_A)
        fake_A_xz = torch.transpose(fake_A_xy, 0, 2);
        
        fake_A_xyz = netG_A_stacking(fake_A_xz)
        pred_fake_xz = netD_A_xz(fake_A_xyz)
        loss_GAN_B2A_xz = criterion_GAN(pred_fake_xz, target_real)
        
        fake_A_xyz_xy = torch.transpose(fake_A_xyz, 0, 2);        
        # Pixel consisent loss
        loss_pixel = criterion_identity(fake_A_xyz_xy,fake_A_xy) * 1
        
        loss_G = (loss_GAN_B2A_xz * 1 + loss_pixel)
        loss_G.backward()
        optimizer_G_stack.step()
        
        ###################################
        ###### Discriminator A ######
        optimizer_D_A_xz.zero_grad()
        # Real loss
        pred_real_xz = netD_A_xz(real_A_xz)
        loss_D_real_xz = criterion_GAN(pred_real_xz, target_real)

        fake_A_xyz = fake_A_xyz_buffer.push_and_pop(fake_A_xyz)
       
        pred_fake_xz = netD_A_xz(fake_A_xyz.detach())       
        loss_D_fake_xz = criterion_GAN(pred_fake_xz, target_fake)
        
        loss_D_A_xyz = (loss_D_real_xz + loss_D_fake_xz)*0.5
        
        loss_D_A_xyz.backward()

        optimizer_D_A_xz.step()

        sys.stdout.write("\r[Epoch%d/%d], [Batch%d/%d], [loss_G:%f], [loss_pixel:%f],[loss_D_A_xz:%f]" %
                           (epoch+1, opt.n_epochs,
                           i, len(train_loader),
                           loss_G.data.cpu(),
                            loss_pixel.data.cpu(),
                           (loss_D_A_xyz).data.cpu(),
                           ))

    print('Epoch:', epoch+1, '| Learning rate_G', optimizer_G_stack.state_dict()['param_groups'][0]['lr'])
    
    # Update learning rates
    lr_scheduler_G_stack.step()
    lr_scheduler_D_A_xz.step()

    # Save models checkpoints
    PATH = os.path.join(opt.dir_checkpoint, '{}.pt'.format(epoch + 1))
    torch.save({
                'epoch': epoch,
                'netG_A_stacking_state_dict': netG_A_stacking.state_dict(),
                }, PATH)
    print("==============Checkpoint saved==============")
    #%%
    name = listdir(opt.dir_test_data)
    B = io.imread(os.path.join(opt.dir_test_data, name[1]))
    im_tfs = tfs.Compose([
          tfs.ToTensor(),
          ])
    B = im_tfs(B)
    B = torch.transpose(B, 2, 0); 
    B = B.unsqueeze(1)
    with torch.no_grad():
        B2A = netG_B2A1(B.cuda(opt.gpu))  
    B1 = torch.transpose(B2A, 0, 2); 
    
    checkpoint = torch.load(os.path.join(opt.dir_checkpoint, '{}.pt'.format(epoch + 1)),weights_only=True)
    netG_A_stacking1.load_state_dict(checkpoint['netG_A_stacking_state_dict'])
    netG_A_stacking1.cuda(opt.gpu)
    
    
    with torch.no_grad():
        Bxyz = netG_A_stacking1(B1.cuda(opt.gpu))
        
    out1 = torch.transpose(Bxyz,0,2)
    out1  = out1.clamp(min=0, max=1)
    fake_A_xyz= np.uint8(np.squeeze(out1).cpu().detach().numpy()*255.0)
    
    B2A  = B2A.clamp(min=0, max=1)
    fake_A_xy = np.uint8(np.squeeze(B2A).cpu().detach().numpy()*255.0)

    if k == 0:
        path = os.path.join(opt.dir_test_data, f'With_Misalignment.tif')
        io.imsave(path, fake_A_xy)
    
    path = os.path.join(opt.dir_test_data, f'Misalignment fixed/nF0_{k:02d}.tif')
    io.imsave(path, fake_A_xyz)
    k += 1
    
    
    
    
    
    
    
    
    
