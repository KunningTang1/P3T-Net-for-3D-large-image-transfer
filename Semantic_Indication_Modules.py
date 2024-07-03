import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from datetime import datetime

def unet_conv(in_planes, out_planes):
    conv = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(False),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(False),
    )
    return conv
class Uresnet(nn.Module):
    def __init__(self, input_nbr = 1,label_nbr = 3, block = 28):
        super(Uresnet, self).__init__()
        
        # forward
        self.downconv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        ) 
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
        )      # No1 resudual block
        
        self.downconv3 = unet_conv(128, 128) # No2 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3,1,1),
            nn.ReLU(True),
        )      # No2 resudual block
        
        self.downconv5 = unet_conv(256, 256) # No3 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3,1,1),
            nn.ReLU(True),
        )      # No3 resudual block
        
        self.downconv7 = unet_conv(512, 512) # No4 long skip
        
        
        self.updeconv2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
           
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv4 = unet_conv(256, 256)
        
        self.updeconv3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
           
        self.upconv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv6 = unet_conv(128, 128)
        self.updeconv4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        self.upconv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(False),   
        )       # No6 resudual block
        self.upconv8 = unet_conv(64, 64)
        
        self.last = nn.Conv2d(64, label_nbr, 1)  # 6 is number of phases to be segmented
        
        self.fc_params = nn.Sequential (
            nn.Linear(512*block*block, 512),
            nn.BatchNorm1d(512),
            )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 10)
            )
        
        self.fc_paramsShallow = nn.Sequential (
            nn.Linear(128*block*block, 128),
            nn.BatchNorm1d(128),
            )

        self.classifierShallow = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10)
            )

    def forward(self, x): 
        # encoding
        x1 = self.downconv1(x) 
 
        x2 = self.maxpool(x1)     
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)      
        x4 = x3 + x4
        x5 = self.maxpool(x4)
        
        x6 = self.downconv4(x5)
        x7 = self.downconv5(x6)
        x7 = x6 + x7
        x8 = self.maxpool(x7)
        
        x9 = self.downconv6(x8)
        x10 = self.downconv7(x9)
        x10 = x9 + x10
      
        y3 = nn.functional.interpolate(x10, mode='bilinear', scale_factor=2, align_corners=True)
        y4 = self.updeconv2(y3)
        y5 = self.upconv3(torch.cat([y4, x7],1))
        y6 = self.upconv4(y5)
        y6 = y5 + y6
        
        y6 = nn.functional.interpolate(y6, mode='bilinear', scale_factor=2, align_corners=True)
        y7 = self.updeconv3(y6)   
        y8 = self.upconv5(torch.cat([y7, x4],1))
        y9 = self.upconv6(y8)
        y9 = y8 + y9
        
        y9 = nn.functional.interpolate(y9, mode='bilinear', scale_factor=2, align_corners=True)
        y10= self.updeconv4(y9)
        y11 = self.upconv7(torch.cat([y10, x1],1))
        y12 = self.upconv8(y11)
        y12 = y11 + y12
     
        out = self.last(y12)
        
        full = x10.view(x10.size(0), -1)
        full = self.fc_params(full)
        score = self.classifier(full)
        
        shallow = x10.view(x4.size(0), -1)
        shallow = self.fc_params(shallow)
        scoreshallow = self.classifier(shallow)
        
        return out, score, scoreshallow

def uresnet():
    net = Uresnet()
    return net    

class Uresnet1(nn.Module):
    def __init__(self, input_nbr = 1,label_nbr = 3):
        super(Uresnet1, self).__init__()
        
        # forwarf
        self.downconv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )      # No.1 long skip 
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
        )      # No1 resudual block
        
        self.downconv3 = unet_conv(128, 128) # No2 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3,1,1),
            nn.ReLU(True),
        )      # No2 resudual block
        
        self.downconv5 = unet_conv(256, 256) # No3 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3,1,1),
            nn.ReLU(True),
        )      # No3 resudual block
        
        self.downconv7 = unet_conv(512, 512) # No4 long skip
        
        
        self.updeconv2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
           
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv4 = unet_conv(256, 256)
        
        self.updeconv3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
           
        self.upconv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv6 = unet_conv(128, 128)
        self.updeconv4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        self.upconv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(False),   
        )       # No6 resudual block
        self.upconv8 = unet_conv(64, 64)
        
        self.last = nn.Conv2d(64, label_nbr, 1)  # 6 is number of phases to be segmented

    def forward(self, x):
        
        # encoding
        x1 = self.downconv1(x) 
 
        x2 = self.maxpool(x1)     
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)      
        x4 = x3+ x4
        x5 = self.maxpool(x4)
        
        x6 = self.downconv4(x5)
        x7 = self.downconv5(x6)
        x7 = x6 + x7
        x8 = self.maxpool(x7)
        
        x9 = self.downconv6(x8)
        x10 = self.downconv7(x9)
        x10 = x9 + x10
      
        y3 = nn.functional.interpolate(x10, mode='bilinear', scale_factor=2, align_corners=True)
        y4 = self.updeconv2(y3)
        y5 = self.upconv3(torch.cat([y4, x7],1))
        y6 = self.upconv4(y5)
        y6 = y5 + y6
        
        y6 = nn.functional.interpolate(y6, mode='bilinear', scale_factor=2, align_corners=True)
        y7 = self.updeconv3(y6)   
        y8 = self.upconv5(torch.cat([y7, x4],1))
        y9 = self.upconv6(y8)
        y9 = y8 + y9
        
        y9 = nn.functional.interpolate(y9, mode='bilinear', scale_factor=2, align_corners=True)
        y10= self.updeconv4(y9)
        y11 = self.upconv7(torch.cat([y10, x1],1))
        y12 = self.upconv8(y11)
        y12 = y11 + y12
     
        out = self.last(y12)
        
        return out
def uresnet1():
    net = Uresnet1()
    return net 


if __name__ == "__main__":  
    colormap = [[1,1,1],[2,2,2],[3,3,3]]
    net = Uresnet()
    # print(net)
    if torch.cuda.is_available():
        net = net.cuda()    
    num_classes = len(colormap)
    cm2lbl = np.zeros(256**3)
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
    
    def image2label(im):
        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :,1]) * 256 + data[:, :,2]
        return np.array(cm2lbl[idx], dtype='int64')


    ROOT = "H:\\DomainTransfer\\Case4\\Training2DCycSem_TT\\"
    # Reading image data
    def read_image(mode="train", val=False):
        if(mode=="train"): 
            filename = ROOT + "\\train.txt"
        elif(mode == "test"): 
            filename = ROOT + "\\test.txt"
        data = []
        label = []
        with open(filename, "r") as f:
            images = f.read().split()
            for i in range(len(images)):
                if(i%2 == 0):
                    data.append(ROOT+images[i])
                else:
                    label.append(ROOT+images[i])

        print(mode+":contains: "+str(len(data))+" images")
        print(mode+":contains: "+str(len(label))+" labels")
        return data, label
    
    
    data, label = read_image("train")
       
    size = 224
    def crop(data, label, height=size, width=size):
        st_x = 0
        st_y = 0
        box = (st_x, st_y, st_x+width, st_y+height)
        data = data.crop(box)
        label = label.crop(box)
        return data, label
    
    def image_transforms(data, label, height=size, width=size):
        data, label = crop(data, label, height, width)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),
        ])
        
        data = im_tfs(data)
        label = np.array(label)
        label = image2label(label)
        label = torch.from_numpy(label).long()
        return data, label
    
    class SegmentDataset(torch.utils.data.Dataset):
        
        # make functions
        def __init__(self, mode="train", height=size, width=size, transforms=image_transforms):
            self.height = height
            self.width = width
            self.transforms = transforms
            data_list, label_list = read_image(mode=mode)
            self.data_list = data_list
            self.label_list = label_list
            
        
        # do literation
        def __getitem__(self, idx):
            img = self.data_list[idx]
            label = self.label_list[idx]
            img=Image.open(img)
            label=Image.open(label)
            label= label.convert('RGB')
            img, label = self.transforms(img, label, self.height, self.width)
            return img, label
        
        def __len__(self):
            return len(self.data_list)
    
    
    height = size
    width = size
    Segment_train = SegmentDataset(mode="train")
    Segment_test = SegmentDataset(mode="test")
    
    train_data = DataLoader(Segment_train, batch_size= 20, shuffle=True)
    test_data = DataLoader(Segment_test, batch_size=4)
    
    # Confusion matrix
    def _fast_hist(label_true, label_pred, n_class):
    
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist
    
    """
    label_trues: right label
    label_preds: predic label
    n_class: number of classes
    """
    def label_accuracy_score(label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = np.zeros((n_class, n_class))
    
        for lt, lp in zip(label_trues, label_preds):
            # numpy.ndarray.flatten : numpy to 1D
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        
        # np.diag(a)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)  # nanmean ignore NaN
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc    

    
    LEARNING_RATE = 0.00001
    
    weight_p, bias_p = [],[]
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p] 
        else:
            weight_p += [p]
            
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5 , patience=4, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    criterion = nn.CrossEntropyLoss()
 
    EPOCH = 50
    
    # train data record
    train_loss = []
    train_acc = []
    train_acc_cls = []
    train_mean_iu = []
    train_fwavacc = []
    average_train_acc = []
    # valid data record
    k = 0

    for epoch in range(EPOCH):
        _train_loss = 0
        _train_acc = 0
        _train_acc_cls = 0
        _train_mean_iu = 0
        _train_fwavacc = 0
        _each_acc_train = 0
        prev_time = datetime.now()
        net = net.train()
        for step, (x,label) in enumerate(train_data):
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
                      
            out,_,_= net(x)
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
        train_acc.append(_train_acc/len(Segment_train))
        average_train_acc.append(_each_acc_train/len(Segment_train))
        scheduler.step(_train_loss/len(train_data))
        epoch_str = ('Epoch: {}, train Loss: {:.5f}, train Weight Acc: {:.5f}, train UNWeight Acc: {:.5f} '.format(
            epoch+1, _train_loss / len(train_data), _train_acc / len(Segment_train), _train_mean_iu / len(Segment_train)))
        print(epoch_str)
        print('')
        print('Epoch:', epoch+1, '| Learning rate_D', optimizer.state_dict()['param_groups'][0]['lr'])

        if epoch == k:
            PATH = 'H:\\DomainTransfer\\Case4\\checkpoints\\ck_pretrainSegnet_TT\\%d' % (epoch+1) +'.pt'
            torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
            k += 1


