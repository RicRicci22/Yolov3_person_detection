# A Yolov3-based person detection system for search and rescue operations in complex backgrounds
A repository to keep track of the work for my master degree

Next steps:
- [X] correct precision recall calculation 
- [X] small medium large objects precision recall calculation 
- [X] insert calculation for AP and AR
- [X] insert calculation for AP small medium large and the same for AR (even tough it does not have much sense maybe..)
- [X] insert validation using training
- [X] implementing loss validation during training
- [X] create the new "per frame" metric 
- [ ] create new visdrone dataset with images containing no person (balanced between containing or not) 
- [X] jetson nano developer kit (https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [X] fine tuning (partendo da pesi di yolov4, freezing) 

Tests:
- [ ] customizing anchor boxes 
- [ ] adding input size variation during training (see if it helps) -> changed with cropping during data aug.
- [ ] adding data augmentation
- [ ] train keeping aspect ratio 


Finetuning method: 
- (1) Begin always from the pretrained yolov4 weights. 
- (2) Freeze backbone and neck and train only the heads for 15 epochs with learning rate 0.001
- (3) Unfreeze backbone and neck and train on the complete network for 10 epochs with learning rate 0.0005
- (4) Freeze again neck and backbone and fine tune head for 5 epochs with learning rate 0.0005

Tasks completed (05/09/2021):
- [X] create another custom dataset, with also negative examples
- [X] created automatic finetuning method 

Tasks completed (07/09/2021):
- [X] add augmentation strategy of cropping (to use different scales without changing the input resolution)
- [X] add augmentation strategy of rotating (+90 -90) can help in lying position
- [X] write the section about python and pytorch 
- [X] check data augmentation strategy 
- [X] write activation functions 

Tasks completed (16/09/2021):
- [X] print anchor box activation during training.
- [X] correct the small medium large calculation by reffering to percentage over the image 
- [X] finish benchmarks
- [X] calculate target size number on trainsets

Tomorrow:
- [X] Add examples images in data augmentation 
- [ ] Insert all results in test
- [ ] Correct introduction and yolo paragraphs 
- [X] Create jetson nano launcher
- [ ] Correct FPS and metrics values in test resolutions 
- [ ] Create onnx launcher ? 
- [ ] Insert images of custom dataset 


AUGMENTATION STRATEGIES 
- gaussian noise -> random value between 0.01 and 0.001. Set to true or false 
- blur -> shall be a small value, since there are small objects 
- mosaic -> fuse 4 images in one, by taking pieces of each one
- flip -> randomly flip image on x, y or both axes 
- rotate -> randomly rotate image by +- 90 degrees. Can be useful for people in lying position? 
