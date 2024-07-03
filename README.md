**PET-Net — PyTorch implementation

This is the source code for P3T-Net designed for 3D image-to-image transfer. P3T-Net is built purely in 2D but can work on 3D images. 
The main application of this network is to process images of materials taken from 3D imaging techniques, such as micro-CT and nano-CT. 
The objective of this network is to transfer a domain into another domain in an unparied manner, in terms of grayscale distribution, image style, and image semantics.
P3T-Net is very computationally efficient. For a typical size of 3D micro-CT and nano-CT images at 1000^3 to 3000^3, it can handle these images in a local-degree workstation with a GPU of 24GB memory.

Here are two examples of using P3T-Net to transfer a dynamic scan micro-CT image of sandstone to a very high-quality image:
![F1](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/8a2c9498-2779-4e3f-a9eb-a3fd354060ce)
![F2](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/2b162de5-90a3-4d73-a9cb-bbb895be0c4c)

Here is an example of using P3T-Net to transfer a fast scan micro-CT image of a fuel cell to a very high-quality image:
![F3](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/d29f8c01-2265-4e34-bbe2-8c4155eb6385)

**Requirements

Linux and Windows are supported, but the presented code is implemented under Windows.

NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA RTX4090 with 24GB of memory.

64-bit Python 3.8.10 and PyTorch 2.3.1. See https://pytorch.org/ for PyTorch install instructions.

CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 4090.

Python image processing libraries are required, which are scikit-image 0.17.2 and Pillow 3.10.1.


**Preparing datasets

Two 3D images are required to train the P3T-Net, one serves as the source domain, and another serves as the target domain. 

Example datasets for demonstration can be found at https://zenodo.org/records/12630142. 

This dataset includes a nano-CT image of a dual-mode scan of a Lithium-ion battery cathode (voxel size: 128nm); and a nano-CT image of a single-mode scan of a lithium-ion battery cathode (voxel size: 128nm).



