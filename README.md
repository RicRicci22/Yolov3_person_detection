# master_degree

A repository to keep track of the work for my master degree.

STEPS:
- Set up yolov4 on colab and create a python file that do the same thing locally (to use in laboratory afterwards) 
- Process the sample video using non fine-tuned yolov4 to test benchmark performance (track also the timings). Process the video with different FPS to compare results 
- Fine tune yolov4 using VisDrone/sardCorr1 datasets and repeat the processing on the video performing the same tests to compare them
- Reproduce the results of the paper () using the fine tuned yolov4
- Perform the same steps as above but using a two-stages approach

Substeps: 
- Improve the video output -> it takes too much space now
- Try with different resolutions and frame skip
- Write the time it takes to process a frame


UPDATE 14/06/2021

I'm currently trying to set up GPU to use yolov4 following this guide https://dsbyprateekg.blogspot.com/2020/05/how-to-install-and-compile-yolo-v4-with.html.
First step is to install CUDA 10.1 and cudnn 7.6.4 on the machine. 
The machine has a TESLA K40c GPU. 
Now I'm installing CUDA 10.1, let's see if after that it's needed to edit the system variables. After that I will look inside the installing of CuDNN, but on the pc of farid there are already two folders containing CuDNN, let's see if the version is compatible, otherwise I will have to remove it and install the correct one.
******
Installed CUDA 10.1
******
Installed CuDNN 7.6.5 for CUDA 10.1
******

I deleted CUDA_VERSION_10.0 PATH from environment paths, now it seems to compile darknet correctly using vcpkg. 
The idea is to find a way to train the custom yolov4 on google colab, and then use the weights to perform detection on the laboratory machine. In this way we can have a realistic sense of performance, as well as learning how to compile yolo on the drone.
