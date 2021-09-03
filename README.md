# A Yolov4-based person detection system for search and rescue operations in complex backgrounds
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
- [ ] jetson nano developer kit (https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [ ] fine tuning (partendo da pesi di yolov4, freezing) 

Tests:
- [ ] customizing anchor boxes 
- [ ] adding a detection layer if needed 
- [ ] adding input size variation during training (see if it helps) 
- [ ] adding data augmentation
- [ ] train keeping aspect ratio 
