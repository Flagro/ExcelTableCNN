# ExcelTableCNN
This repository is for ExcelTableCNN project - open source automatic table detection on Excel sheets with computer vision.

The project's goal is to implement OpenSource library for easy and convenient detection of multiple Table Header/Data coordinates by the use of algorithms combined with Deep Learning/Computer Vision approaches.

## The model design
The first version of the ExcelTableCNN would try to recreate the [TableSense model](https://arxiv.org/abs/2106.13500). So the model structure would look like this:
1) DataLoader with minibatch=1
2) FCN backbone (this is where the number of channels would be normalized to be 3)
3) RPN - generationg of RoI
4) For each RoI RoIAlign would be implemented and for each RoIAlign segment would be done
    - Table classification - change of table presence in the RoI
    - BBR + TableSense smart RoIAlign for each bounding box prediction
    - Table segmentation - binary mask for table present in each cell of RoI
5) For each RoI bounding boxes would be ranked and filtered with NMS

## Performance Metrics 
TODO

## Training Data
For this project the data that's going to be used is based on VEnron2 dataset with markup from [TableSense project](https://github.com/microsoft/TableSense/blob/main/dataset/Table%20range%20annotations.txt).
