import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import cv2
import random
from os import listdir

#%% Load training volumes
path_data = '../Testingdata'
num_volumes = listdir(path_data)
sd = io.imread(os.path.join(path_data, num_volumes[0]))
td = io.imread(os.path.join(path_data, num_volumes[1]))
td_lb = io.imread(os.path.join(path_data, num_volumes[2]))

plt.subplot(131)
plt.imshow(sd[100,:,:],cmap = 'gray');plt.title('Source domain')
plt.subplot(132)
plt.imshow(td[100,:,:],cmap = 'gray');plt.title('Unpaired Target domain')
plt.subplot(133)
plt.imshow(td_lb[100,:,:]);plt.title('Target domain segmentation')

if len(td_lb.shape) == 3:
    classes = np.unique(td_lb)
print('Image contains: ', classes, "labels")
#%% 
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)  
directories = [
    "../Example/Semantic Indication Module",
    "../Example/Semantic Indication Module/checkpoints",
    "../Example/Semantic Indication Module/Train/trainA",
    "../Example/Semantic Indication Module/Train/trainB",
    "../Example/Domain Transfer Module",
    "../Example/Domain Transfer Module/checkpoints",
    "../Example/Domain Transfer Module/Train/trainA",
    "../Example/Domain Transfer Module/Train/trainB",
    "../Example/Domain Transfer Module/Train/trainsegA",
    "../Example/Domain Transfer Module/Test",
    "../Example/Domain Transfer Module/Test/Fake_target",
    "../Example/Domain Transfer Module/Test/Fake_source",
    "../Example/Misalignment fixing module",
    "../Example/Misalignment fixing module/checkpoints",
    "../Example/Misalignment fixing module/Train/trainA",
    "../Example/Misalignment fixing module/Train/trainB",
    "../Example/Misalignment fixing module/Test/Misalignment fixed",
]

# Iterate over the list and create directories
for directory in directories:
    mkdir(directory)


#%% Random crop for semantic indication module
training_patch_size = 224
h, w = training_patch_size,training_patch_size
id = 0
for i in range(0,np.size(td,0),2):
    t = td[i,:,:]
    t_lb = td_lb[i,:,:]  
    count=0
    while 1: 
        y = random.randint(0, np.size(t,0)-training_patch_size)
        x = random.randint(0, np.size(t,1)-training_patch_size)
        gray = t[(y):(y + h), (x):(x + w)]
        lb = t_lb[(y):(y + h), (x):(x + w)]
        cv2.imwrite(directories[2] +'\\'+ str(id) + '.png', gray)
        io.imsave(directories[3] +'\\'+ str(id) + '.png', lb)
        count+=1
        id +=1
        if count == 6:
            break
print('Overall', id, 'patches has been generated for training semantic indication module') 



#%% Random crop for semantic indication module
index = min(np.size(sd,0),np.size(td,0))
id = 0
for i in range(0,index,2):
    s = sd[i,:,:]
    t = td[i,:,:]
    t_lb = td_lb[i,:,:]  
    count=0
    while 1: 
        y = random.randint(0, np.size(t,0)-training_patch_size)
        x = random.randint(0, np.size(t,1)-training_patch_size)
        gray_s = s[(y):(y + h), (x):(x + w)]
        gray_t = t[(y):(y + h), (x):(x + w)]
        lb = t_lb[(y):(y + h), (x):(x + w)]
        cv2.imwrite(directories[6] +'\\'+ str(id) + '.png', gray_t)
        cv2.imwrite(directories[7] +'\\'+ str(id) + '.png', gray_s)
        io.imsave(directories[8] +'\\'+ str(id) + '.png', lb)
        count+=1
        id +=1
        if count == 10:
            break
    if id >= 3000:
        break
print('Overall', id, 'patches has been generated for training semantic-consistent domain transfer module') 

# Testing image prepare
io.imsave(directories[9]+'\\test_target.png',td[491,:,:])
io.imsave(directories[9]+'\\test_source.png',sd[491,:,:])
#%% Random crop for semantic indication module
index1 = min(np.size(sd,0),np.size(td,0))
index2 = min(np.size(sd,1),np.size(td,1))
index3 = min(np.size(sd,2),np.size(td,2))
size = 48
num = 0
for z in range(size,index3-size,40):
    for y in range(size,index2-size,40):
        for x in range(size,index1-size,40):
            if num<1000:
                c1 = td[z-size:z+size,y-size:y+size,x-size:x+size]               
                n1 = sd[z-size:z+size,y-size:y+size,x-size:x+size]

                io.imsave(directories[14] +'\\'+str(num)+'.tif',c1)
                io.imsave(directories[15] +'\\'+str(num)+'.tif',n1)
                num+=1
print('Overall', num, 'patches has been generated for training misalignment fixing module') 

# Testing image prepare
output_path = os.path.abspath(os.path.join(directories[16], '../test_3Dvolume.tif'))
io.imsave(output_path, sd[70:70+96, 150:150+256, 70:70+256])

