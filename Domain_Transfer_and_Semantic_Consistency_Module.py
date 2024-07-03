import argparse
import itertools
import sys
from torch.utils.data import DataLoader
from PIL import Image
import torch
from Semantic_Indication_Modules import uresnet
from models import Discriminator,FeatureDiscriminator,ResnetGenerator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal2d
import numpy as np
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from torchvision.transforms import Compose, ToTensor

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--dir_train_data', type=str, default='H:\\DomainTransfer\\Case4\\Training2DCycSem_TT\\Train\\',help='dataset directory')
parser.add_argument('--dir_test_data', type=str, default='H:\\DomainTransfer\\Case4\\Training2DCycSem_TT\\Train\\',help='dataset directory')
opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
 #%% Load label of the segmented image
colormap = [[1,1,1],[2,2,2],[3,3,3]]
cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i 

def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') 

def image_transforms(label):
    label = np.array(label)
    label = image2label(label)
    label = torch.from_numpy(label).long() 
    return label
     #%% dataloader
class TrainDatasetFromFolder(Dataset):
    def __init__(self, root, transforms1=image_transforms):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames1 = [join(root, x) for x in listdir(root)][0]
        self.image_filenames2 = [join(root, x) for x in listdir(root)][1]
        self.image_filenames3 = [join(root, x) for x in listdir(root)][2]
        LIST1 = listdir(self.image_filenames1)
        LIST2 = listdir(self.image_filenames2)
        LIST3 = listdir(self.image_filenames3)
        LIST1.sort(key=lambda x: int(x[:-4]))
        LIST2.sort(key=lambda x: int(x[:-4]))
        LIST3.sort(key=lambda x: int(x[:-4]))
        self.img1 = [join(self.image_filenames1, x) for x in LIST1]
        self.img2 = [join(self.image_filenames2, x) for x in LIST2]
        self.img3 = [join(self.image_filenames3, x) for x in LIST3]
        self.train_input_transform = transform()
        self.train_output_transform = transform()
        self.transforms = transforms1
        
    def __getitem__(self, index):
        input_image = self.train_input_transform(Image.open(self.img1[index]))
        output_image = self.train_output_transform(Image.open(self.img2[index]))
        # seg_image= self.transforms((Image.open(self.img3[index])).convert('RGB'))
        label= Image.open(self.img3[index]).convert('RGB')
        seg_image = self.transforms(label)
        return input_image,output_image,seg_image

    def __len__(self):
        return len(self.img1)
def transform():
    return Compose([
        ToTensor()       
    ])

class TestDatasetFromFolder(Dataset):
    def __init__(self, root, transforms1=image_transforms):
        super(TestDatasetFromFolder, self).__init__()
        self.transforms = transforms1
        self.image_filenames1 = [join(root, x) for x in listdir(root)][0]
        self.image_filenames2 = [join(root, x) for x in listdir(root)][1]
        self.image_filenames3 = [join(root, x) for x in listdir(root)][2]
        LIST1 = listdir(self.image_filenames1)
        LIST2 = listdir(self.image_filenames2)
        LIST3 = listdir(self.image_filenames3)
        LIST1.sort(key=lambda x: int(x[:-4]))
        LIST2.sort(key=lambda x: int(x[:-4]))
        LIST3.sort(key=lambda x: int(x[:-4]))
        self.img1 = [join(self.image_filenames1, x) for x in LIST1]
        self.img2 = [join(self.image_filenames2, x) for x in LIST2]
        self.img3 = [join(self.image_filenames3, x) for x in LIST3]
        self.test_input_transform = transform()
        self.test_output_transform = transform()

    def __getitem__(self, index):
        input_image = self.test_input_transform(Image.open(self.img1[index])) #lr input
        output_image = self.test_output_transform(Image.open(self.img2[index]))#clr
        label= Image.open(self.img3[index]).convert('RGB')
        seg_image = self.transforms(label)
        return input_image,output_image,seg_image
    def __len__(self):
        return len(self.img1) 
    
###### Definition of variables ######
 #%% Networks
netG_A2B1 =  ResnetGenerator(opt.input_nc, opt.output_nc,n_blocks= 4)
netG_B2A1 =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks= 4)
netG_A2B =  ResnetGenerator(opt.input_nc, opt.output_nc,n_blocks= 4)
netG_B2A =  ResnetGenerator(opt.output_nc, opt.input_nc,n_blocks= 4)
netD_A = Discriminator()
netD_B = Discriminator()

netD_fs = FeatureDiscriminator()
netD_fs_shallow = FeatureDiscriminator()

 #%% load pretrainied Segnet and freeze
netfs = uresnet()
checkpoint = torch.load('H:\\DomainTransfer\\Case4\\checkpoints\\ck_pretrainSegnet_TT\\13.pt')
netfs.load_state_dict(checkpoint['net_state_dict'])

for param in netfs.parameters():
    param.requires_grad = False
    
netD_fs_shallow.cuda()
netfs.cuda()
netD_fs.cuda()
netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()

netG_A2B.apply(weights_init_normal2d)
netG_B2A.apply(weights_init_normal2d)
netD_A.apply(weights_init_normal2d)
netD_B.apply(weights_init_normal2d)
netD_fs.apply(weights_init_normal2d)
netD_fs_shallow.apply(weights_init_normal2d)
 #%% Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_seg1 =torch.nn.L1Loss()
criterion_seg = torch.nn.modules.CrossEntropyLoss()
criterion_fsgan =torch.nn.MSELoss()
criterion_fsgan_sl =torch.nn.MSELoss()

 #%% Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_fs = torch.optim.Adam(netD_fs.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_fs_sl = torch.optim.Adam(netD_fs_shallow.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_fs = torch.optim.lr_scheduler.LambdaLR(optimizer_D_fs, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_fs_sl = torch.optim.lr_scheduler.LambdaLR(optimizer_D_fs_sl, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
 #%% Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
fake_A_seg_buffer = ReplayBuffer()
fake_A_sl_buffer = ReplayBuffer()

 #%% Dataset loader
train_set = TrainDatasetFromFolder(opt.dir_train_data)
test_set = TestDatasetFromFolder(opt.dir_test_data)
train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=opt.batchSize, shuffle=False)
###################################
k = 0
###### Training ######
netfs.eval()
for epoch in range(opt.epoch, opt.n_epochs):
    for i,  (real_A, real_B,gt_A) in enumerate(train_loader):
        real_A = real_A.cuda()
        real_B = real_B.cuda()
        gt_A = gt_A.cuda()

        source_label = torch.ones(real_A.size(0),2, requires_grad=False).cuda()
        target_label = torch.zeros(real_A.size(0),2, requires_grad=False).cuda()
        target_real = torch.ones(real_A.size(0), requires_grad=False).cuda()
        target_fake = torch.zeros(real_A.size(0), requires_grad=False).cuda()
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        
        #%$==================================================================================
        ### Semantic loss
        fake_B_seg,feature_fakeB,_ = netfs(fake_B)
        real_A_seg,feature_realA,shallow_realA = netfs(real_A)   
        loss_seg_A = criterion_seg1(feature_fakeB,feature_realA)

        fake_A_seg,feature_fakeA,shallow_fakeA = netfs(fake_A)
        real_B_seg,feature_realB,_ = netfs(real_B)
        loss_seg_B = criterion_seg1(feature_fakeA,feature_realB)
        
        loss_seg = loss_seg_A+ loss_seg_B
        
        ### Semantic 2
        recovered_A_seg,_,_ = netfs(recovered_A)
        loss_seg_recovered_A = criterion_seg(recovered_A_seg,gt_A)
           
        fs_fake_A = netD_fs(feature_fakeA)
        loss_GAN_fake_A = criterion_fsgan(fs_fake_A,source_label)
        
        #%$==================================================================================
        # Total loss
        loss_G = loss_identity_A + loss_identity_B + (loss_GAN_A2B + loss_GAN_B2A) * 0.8 + loss_cycle_ABA+ loss_cycle_BAB + (loss_seg+loss_seg_recovered_A+loss_GAN_fake_A)*0.8
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ##################################
        ##### Discriminator fs ######
        optimizer_D_fs.zero_grad()

        fs_gt_A = netD_fs(feature_realA)
        loss_GAN_gt_A = criterion_fsgan(fs_gt_A,source_label)
        
        feature_fakeA  = fake_A_seg_buffer.push_and_pop(feature_fakeA)

        
        fs_fake_A = netD_fs(feature_fakeA.detach())
        loss_GAN_fakeA_D = criterion_fsgan(fs_fake_A,target_label)
        
        
        loss_fs = (loss_GAN_fakeA_D + loss_GAN_gt_A)*0.5
        loss_fs.backward()

        optimizer_D_fs.step()
        #%%
        sys.stdout.write("\r[Epoch%d/%d], [Batch%d/%d], [loss_G:%f], [loss_G_GAN:%f], [loss_G_cycleABA:%f],[loss_G_cycleBAB:%f], [semanticTotal:%f],[semantic_recoverA:%f], [loss_D_A:%f], [loss_D_B:%f],[loss_D_fs:%f]" %
                                        (epoch+1, opt.n_epochs,
                                        i, len(train_loader),
                                        loss_G.data.cpu(),
                                        (loss_GAN_A2B + loss_GAN_B2A).data.cpu(), loss_cycle_ABA.data.cpu(),loss_cycle_BAB.data.cpu(),
                                        (loss_seg).data.cpu(),loss_seg_recovered_A.data.cpu(),
                                        (loss_D_A).data.cpu(),(loss_D_B).data.cpu(),loss_fs.data.cpu()
                                        ))
        
    print('Epoch:', epoch+1, '| Learning rate_G', optimizer_G.state_dict()['param_groups'][0]['lr'])
    print('Epoch:', epoch+1, '| Learning rate_G', optimizer_D_A.state_dict()['param_groups'][0]['lr'])
    print('Epoch:', epoch+1, '| Learning rate_G', optimizer_D_B.state_dict()['param_groups'][0]['lr'])
    

     #%% Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

     #%% Save models checkpoints
    PATH = 'H:\\DomainTransfer\\Case4\\checkpoints\\ck_CycSem\\%d' % k +'.pt'

    torch.save({
                'epoch': epoch,
                'netG_A2B_state_dict': netG_A2B.state_dict(),
                'netG_B2A_state_dict': netG_B2A.state_dict(),
                }, PATH)
    

    

    
    
    
