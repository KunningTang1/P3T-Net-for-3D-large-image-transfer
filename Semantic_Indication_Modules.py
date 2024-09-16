import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import Uresnet
import argparse
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from utils import label_accuracy_score,weights_init_normal
import os
import gc 
gc.collect()
torch.cuda.empty_cache()
print('============== Start training Semantic Indication Module============== ')
#%% 
# User has to defined this, this value can be read from the trainingdatePrepare.py 
colormap = [[0,0,0],[85,85,85],[171,171,171]]
# colormap = [[0,0,0],[166,111,0],[255,0,0],[255,153,40],[255,192,192],[255,153,40]]
num_classes = len(colormap)

cm2lbl = np.zeros(256**3) # Every pixel is in range of 0 ~ 255，RGB with 3 channels
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UResNet', 
                    help='define which gpu to use, can be "UResNet"')
parser.add_argument('--epoch', type=int, default=0,
                    help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2,
                    help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00003, 
                    help='initial learning rate')
parser.add_argument('--input_nc', type=int, default=1, 
                    help='number of channels of input data')
# parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--checkpoint', type=int, default=5,
                    help='interval of saving checkpoint')
parser.add_argument('--size', type=int, default=224, 
                    help='size of the data crop (squared assumed)')
parser.add_argument('--gpu', type=int, default=0, 
                    help='define which gpu yo use, can be "0" or "1" if you have two gpus')
parser.add_argument('--dir_train_data', type=str, default='../Example/Semantic Indication Module/Train',
                    help='dataset directory')
parser.add_argument('--dir_checkpoint', type=str, default='../Example/Semantic Indication Module/checkpoints',
                    help='dataset directory')
opt = parser.parse_args()
print(opt)

def crop(data, label, height=opt.size, width=opt.size):
    st_x = 0
    st_y = 0
    box = (st_x, st_y, st_x+width, st_y+height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label

def label_transforms(data, label, height=opt.size, width=opt.size):
    data, label = crop(data, label, height, width)
    # convert to tensor, and normalization
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
    ])
    
    data = im_tfs(data)
    label = np.array(label)
    label = image2label(label)
    label = torch.from_numpy(label).long()   # CrossEntropyLoss require a long() type
    return data, label

class TrainDatasetFromFolder(Dataset):
    def __init__(self, root):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames1 = [join(root, x) for x in listdir(root)][0]
        self.image_filenames2 = [join(root, x) for x in listdir(root)][1]
        LIST1 = listdir(self.image_filenames1)
        LIST2 = listdir(self.image_filenames2)
        LIST1.sort(key=lambda x: int(x[:-4]))
        LIST2.sort(key=lambda x: int(x[:-4]))
        self.img1 = [join(self.image_filenames1, x) for x in LIST1] 
        self.img2 = [join(self.image_filenames2, x) for x in LIST2]
        self.label_transform = label_transforms
        self.height = opt.size
        self.width = opt.size
    def __getitem__(self, index):
        image = Image.open(self.img1[index])
        lb = Image.open(self.img2[index]).convert('RGB')
        image, label = self.label_transform(image, lb ,self.height, self.width)
        
        return image,label

    def __len__(self):
        return len(self.img1)

#%%     
if opt.model == 'UResNet':
    net = Uresnet(opt.input_nc,num_classes)   
    print('============== UresNet is used ===================')

if torch.cuda.is_available():   
    net = net.cuda(opt.gpu)          
net.apply(weights_init_normal); 
print('============== Model initialised ===================')

train_set = TrainDatasetFromFolder(opt.dir_train_data)
train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
print("Training data successfully loaded")
print("Training contains: "+str(len(train_set))+" images")
print("Training contains: "+str(len(train_set))+" labels")
            
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5 , patience=6, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# normedWeights = [1,1,1]
# normedWeights = torch.FloatTensor(normedWeights).cuda(0)    
# criterion = nn.CrossEntropyLoss(weight = normedWeights)
criterion = nn.CrossEntropyLoss()
print("============== Adam optimiser and unweighted CrossEntropyloss are used...")

def predict(img, label): # prediction
    img = img.unsqueeze(0).cuda()
    out = net(img)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    return pred, label
# train data record
train_loss = []
train_acc = []
train_acc_cls = []
train_mean_iu = []
train_fwavacc = []
average_train_acc = []
print("==============Start training model......")
for epoch in range(opt.n_epochs):
    _train_loss = 0
    _train_acc = 0
    _train_acc_cls = 0
    _train_mean_iu = 0
    _train_fwavacc = 0
    _each_acc_train = 0
    net = net.train()
    
    for step, (x,label) in enumerate(train_data):
        
        if torch.cuda.is_available():
            x = x.cuda(opt.gpu)
            label = label.cuda(opt.gpu)
            
        out,_ = net(x)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _train_loss += loss.item()

        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            _train_acc += acc
            _train_acc_cls += acc_cls
            _train_mean_iu += mean_iu
            _train_fwavacc += fwavacc
        
    # recold loss and acc in the epoch
    train_loss.append(_train_loss/len(train_data))
    train_acc.append(_train_acc/len(train_set))
    average_train_acc.append(_train_mean_iu/len(train_set))
    
    scheduler.step(_train_loss/len(train_data))
    epoch_str = ('Epoch: {}, train Loss: {:.5f}, train Weight Acc: {:.5f}, train UNWeight Acc: {:.5f} '.format(
        epoch+1, _train_loss / len(train_data), _train_acc / len(train_set), _train_mean_iu / len(train_set)))
    print(epoch_str)
    print('')
    print('Epoch:', epoch+1, '| Learning rate_D', optimizer.state_dict()['param_groups'][0]['lr'])
    #%% save checkpoint
    if epoch == opt.checkpoint:
        PATH = os.path.join(opt.dir_checkpoint, '{}.pt'.format(epoch + 1))
        torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
        print("==============Checkpoint saved==============")
        opt.checkpoint += 2
    print("==============================================")       

# train loss visualization  
plt.figure()  
epoch = np.array(range(opt.n_epochs))
plt.plot(epoch, train_loss, label="train_loss")
plt.title("loss during training")
plt.legend()
plt.grid()
plt.xlabel('Epoch');plt.ylabel('Loss')

plt.figure()  
# train acc/ valid acc visualization    
plt.plot(epoch, train_acc, label="train_acc")
plt.plot(epoch, average_train_acc, label="average_train_acc")
plt.title("accuracy during training")
plt.legend()
plt.grid()
plt.xlabel('Epoch');plt.ylabel('Accuracy')
print("==============Training finished==============")
    

