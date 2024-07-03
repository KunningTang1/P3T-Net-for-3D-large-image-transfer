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

Linux and Windows are supported, but the presented code is implemented under Windows.

NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA RTX4090 with 24GB of memory.

64-bit Python 3.8.10 and PyTorch 2.3.1. See https://pytorch.org/ for PyTorch install instructions.

CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 4090.

Python image processing libraries are required, which are scikit-image 0.17.2, OpenCV 4.4.0.44, and Pillow 3.10.1. We use the Anaconda3 distribution which installs most of these by default.


# Preparing datasets

Two 3D images are required to train the P3T-Net, one serves as the source domain, and another serves as the target domain. 

Example datasets for demonstration can be found at https://zenodo.org/records/12630142. 

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
For processing a whole 3D image at a typical micro/nano-CT scale (2000^3) on a RTX4090, the processing time is around 30mins.  
![F4](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/258a7bb0-b6ee-48df-9b72-0dcece472785)
