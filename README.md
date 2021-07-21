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

*Darknet installation*

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

******
Installed and compiled darknet using command .\vcpkg install darknet[full]:x64-windows.
******
Darknet worked on a sample input video. Average FPS 7.1.

14.50
*Open-cv dnn module with GPU acceleration installation*

I want to try with a different approach, using open-cv dnn library, to see if there are changes in the FPS. Also this can be useful since the ease of use of visual studio make the entire process more straighforward to apply! Let's try with this walktrough https://medium.com/analytics-vidhya/object-detection-on-public-webcam-with-opencv-and-yolov4-9ed51d5896a9

UPDATE 18/06/2021

I managed to have the lab machine to process a video using GPU! With GPU it reaches 4 FPS on "busy" images, while a little more on not so busy images! This I think is due to the NMS that have to deals with all the bounding box proposals. 
The network is still a little slow, but it's an improvement cause it allows me to train on a custom dataset now!
The implementation followed the guide of https://github.com/roboflow-ai/pytorch-YOLOv4 repository and a youtube video https://www.youtube.com/watch?v=9hVgyeI4g4o&t=1154s. 
It uses pytorch 1.4.0, but since this version does not support 3.5 compute capability GPUs I used pytorch 1.2.0 and it worked just fine.
I will try this afterwards, for now the next steps are 
- [x] video with decreasing confidence values (0.5,0.4,0.3,0.2,0.1). 
- [x] only detect people.
- [x] try to use pytorch 1.4.0 in some way to compare the speed. 
- [ ] try with different input sizes 
- [ ] find a way to evaluate the performance (maybe on visdrone or SARD datasets is possible) 

UPDATE 21/06/2021

Made two functions, one to process all the videos in one folder, the other to process a video with different confidence thresholds from 0.5 to 0.1. The FPS are decreasing for decreasing confidence threshold, due to the Non Maximum Suppression algorithm. 
Next step are: 
- [x] find a way to compress video -> done using moviepy
- [x] understand how a dataset is arranged -> VISDRONE IS ARRANGED AS <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
- [x] train yolov4 on a custom dataset (VISDRONE) 
- [ ] evaluate performance after training (quantitatively on the videos and qualitatively on VISDRONE TEST SET)

UPDATE 21/07/2021
Here I am after a month! I made some progress including finetuning on visdrone and formatting visdrone and sard dataset. Still I need to try finetuning on Sard alone and compare performances.
Next steps are: 
  - [ ] parse corr1 dataset into the format for yolov4 pytorch
  - [ ] try finetuning using only sard/corr dataset and compare performance
  - [ ] find a way to compare perfomances (maybe on sard/corr) given that it is more similar to my scenario 
