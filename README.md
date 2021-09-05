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
- [ ] adding a detection layer if needed 
- [ ] adding input size variation during training (see if it helps) -> changed with cropping during data aug.
- [ ] adding data augmentation
- [ ] train keeping aspect ratio 


Finetuning method: 
- (1) Begin always from the pretrained yolov4 weights. 
- (2) Freeze backbone and neck and train only the heads for 10 epochs with learning rate 0.001
- (3) Unfreeze backbone and neck and train on the complete network for 5 epochs with learning rate 0.0005
- (4) Freeze again neck and backbone and fine tune head for 2 epochs with learning rate 0.0005

Tasks completed (05/09/2021):
- [X] create another custom dataset, with also negative examples
- [X] created automatic finetuning method 

Tomorrow:
- [ ] add augmentation strategy of cropping (to use different scales without changing the input resolution)
- [ ] write the section about python and pytorch 
- [ ] check data augmentation strategy 
- [ ] write activation functions 
