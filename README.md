This is the source code for P3T-Net designed for 3D image-to-image transfer. P3T-Net is built purely in 2D but can work on 3D images. 
The main application of this network is to process images of materials taken from 3D imaging techniques, such as micro-CT and nano-CT. 
The objective of this network is to transfer a domain into another domain in an unparied manner, in terms of grayscale distribution, image style, and image semantics.
P3T-Net is very computationally efficient. For a typical size of 3D micro-CT and nano-CT images at 1000^3 to 3000^3, it can handle these images in a local-degree workstation with a GPU of 24GB memory.

Here is an example of using P3T-Net to transfer a dynamic scan micro-CT image to a very high-quality image:
![F1](https://github.com/KunningTang1/P3T-Net-for-3D-large-image-transfer/assets/97938972/8a2c9498-2779-4e3f-a9eb-a3fd354060ce)
