# P3T-Net — PyTorch implementation

This is the source code for P3T-Net designed for 3D image-to-image transfer. P3T-Net is built purely in 2D but can work on 3D images. 
The main application of this network is to process images of materials taken from 3D imaging techniques, such as micro-CT and nano-CT. 
The objective of this network is to transfer a domain into another domain in an unparied manner, in terms of grayscale distribution, image style, and image semantics.
P3T-Net is very computationally efficient. For a typical size of 3D micro-CT and nano-CT images at 1000^3 to 3000^3, it can handle these images in a local-degree workstation with a GPU of 24GB memory.\
The overall architecture for P3T-Net 
![Figure1 (1)](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/81a883e3-3fb3-4df6-a24f-e00faa66a6ea)

# Examples
Here are two examples of using P3T-Net to transfer a dynamic scan micro-CT image of sandstone to a very high-quality image:
![F1](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/8a2c9498-2779-4e3f-a9eb-a3fd354060ce)
![F2](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/2b162de5-90a3-4d73-a9cb-bbb895be0c4c)

Here is an example of using P3T-Net to transfer a fast scan micro-CT image of a fuel cell to a very high-quality image:
![F3](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/d29f8c01-2265-4e34-bbe2-8c4155eb6385)

# Requirements

Linux and Windows are supported, Three versions are provided:

1. Code for Linux users. The required environments and testing data are included within the code package, allowing users to execute the P3T-NET training and inference directly on a Linux system without the need to manually set up the environments after downloading the package.
   
2. Code for Windows users. The required environments and testing data are included within the code package, allowing users to execute the P3T-NET training and inference directly on a Windows system without the need to manually set up the environments after downloading the package.
   
3. Docker images. A docker image is also created for the code, allowing it to be run easily by importing it into a Docker container.

Step to run the code for result reproduction:

1. The example dataset is located in the file "Testingdata".
  
2. Run TrainingDataPrepare.py to create all the files and training images from the example dataset.

3. Run Semantic_Indication_Modules.py to train the first module for image segmentation.

4. Run Test_SemanticModule.py to test the segmentation network (if needed).

5. Run Domain_Transfer_and_Semantic_Consistency_Module.py to train the second module for domain transfer.

6. Run Misalignment_Fixing_Module.py to train the last module for third-axis misalignment fixing.
 
7. Run Inference.py to use all trained modules for prediction.

All the networks are stored in models.py and some functions are in utils.py.

For Linux users, simply run these comments: 

./1run_TrainingDataPrepare.sh

./2run_Semantic_Indication_Modules.sh

./3run_Test_SemanticModule.sh

./4run_Domain_Transfer_and_Semantic_Consistency_Module.sh

./5run_Misalignment_Fixing_Module.sh

./6run_Inference.sh

For Windows users, simply run these comments: 

./1run_TrainingDataPrepare.sh

./2run_Semantic_Indication_Modules.sh

./3run_Test_SemanticModule.sh

./4run_Domain_Transfer_and_Semantic_Consistency_Module.sh

./5run_Misalignment_Fixing_Module.sh

./6run_Inference.sh


For the docker image, make sure the docker and NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) have been installed. After downloading the "docker_images_p3tnet.tar", follow the  comments below to run the code in the docker container:

Step 1: run: sudo cat docker_images.tar | sudo docker import  - p3t-net/test. To load the docker image into the docker

Step 2: run: sudo docker images. To check if the image is successfully loaded.

Step 3: run: sudo docker run --gpus all  -ti --name work1    p3t-net/test:latest bash. You should be able to enter the docker container.

Step 4: run: cd ./root/code. To enter the code directory.

Step 5: Run the code by using: 

./1run_TrainingDataPrepare.sh

./2run_Semantic_Indication_Modules.sh

./3run_Test_SemanticModule.sh

./4run_Domain_Transfer_and_Semantic_Consistency_Module.sh

./5run_Misalignment_Fixing_Module.sh

./6run_Inference.sh

NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA RTX4090 with 24GB of memory.


# Preparing datasets

Two 3D images are required to train the P3T-Net, one serves as the source domain, and another serves as the target domain. 

Example datasets for demonstration can be found at https://zenodo.org/records/12631632, including:
A pre-trained model for the semantic indication module, which uses a U-ResNet architecture, named Semantic_Indication_Modules.pt\
A pre-trained model for the domain transfer module, which uses a CycleGAN-type architecture, named Domain_transfer.pt\
A pre-trained model for misalignment fixing, which uses a GAN-based architecture, named Misalignment_Fixing_Module.pt\
A noisy synchrotron scan of a sandstone image under a core flooding condition is used for testing.

These pre-trained models were trained to transfer the noisy dynamic synchrotron scan to a lab-based slow scan, as shown in the Example Section.\
These three trained models can be loaded into Inference.py, and perform denoising on the provided noisy synchrotron scan

This dataset includes a nano-CT image of a dual-mode scan of a Lithium-ion battery cathode (voxel size: 128nm); and a nano-CT image of a single-mode scan of a lithium-ion battery cathode (voxel size: 128nm).

A step-by-step data preparation.
1. For the target domain image, segment the 8-bit grayscale image into user-defined phases, for the example given, the image is segmented into pore space and solid phase using a threshold value of 121.
2. Crop the grayscale target image and its segmented image into patches and save them in two separate files, run Semantic_Indication_Modules.py for training the Semantic Indication Module using these grayscale patches as input and segmented patches as ground truth, and save the trained model.
3. Crop the same number of patches from the Source domain image with the same size as the target patches and save them into a third file.
4. Run Domain_Transfer_and_Semantic_Consistency_Module.py using the patches from all three files and load the pre-trained Semantic Indication Module for computing semantic consistency. After training, save the trained model.
5. Load the trained domain transfer module, infer on all 2D slices across the X-Y plane, and stack them into a new 3D image. Transpose the new 3D image and the misalignment issue can be observed.
6. Crop 3D patches (We use 64x64x64) from the new 3D image and save them into a new file.
7. Run Misalignment_Fixing_Module.py using this 3D patches.
8. For inference, run the Inference.py file by loading a new 3D image.

This is noted that we are using the Absolute Path in the source code as the images we dealing with are very large (GB-scale per 3D domain). Ensure that the path is changed to your own direction.

# Inference time
For processing a whole 3D image at a typical micro/nano-CT scale (2000^3) on an RTX4090, the processing time is around 30 minutes.  
![F4](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/258a7bb0-b6ee-48df-9b72-0dcece472785)
