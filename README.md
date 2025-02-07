# YOLO v7 Object Detection

## Problem Statement
Train a YOLO v7 Object Detection model on the provided dataset and evaluate its performance using standard metrics. Additionally, compare YOLO v1 and YOLO v7 in terms of architecture, loss function, and computational complexity.

## Dataset
- The dataset consists of images with bounding box annotations for 7 object classes.
- Each image has a corresponding annotation file in the YOLO format:  
  ```
  class-id center-x center-y width height
  ```
  - Values are normalized between 0 and 1.
- Class definitions are available in the `data.yaml` file.

## Training Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Clone the YOLO v7 repository and set up the environment:
   ```bash
   git clone https://github.com/WongKinYiu/yolov7.git
   cd yolov7
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Place the dataset inside the `data/` directory.
   - Ensure `data.yaml` is correctly configured with paths to training and validation sets.

4. Train the YOLO v7 model:
   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov7.pt --device 0
   ```
5. Jupyter source file of my implementation is available in Codes folder

## Evaluation
Evaluate the trained model on the test set using the following metrics:
1. **mAP@0.5IoU** (Mean Average Precision at 0.5 IoU threshold)
2. **mAP@[0.5:0.95]IoU** (Mean Average Precision averaged over 10 IoU thresholds from 0.5 to 0.95)



## Results
- **Quantitative Results:** Report mAP@0.5IoU and mAP@[0.5:0.95]IoU scores.
- **Qualitative Results:** Visualize sample detections
<img width="829" alt="image" src="https://github.com/user-attachments/assets/900ac8ca-562f-40cb-b3fe-b46fe9bbac02" />
<img width="561" alt="image" src="https://github.com/user-attachments/assets/e56ed8f8-4922-4670-8dae-70f4aca1a9dc" />
<img width="481" alt="image" src="https://github.com/user-attachments/assets/abe5c297-ff79-4c0a-b9f3-2bf117782c43" />

<p align="center">
  <img width="449" alt="image1" src="https://github.com/user-attachments/assets/52bedcc2-075a-47ba-9f0c-349f3acaae32" />
  <img width="446" alt="image2" src="https://github.com/user-attachments/assets/36a40228-3799-4f86-9434-42f9bdfee203" />
</p>

<p align="center">
<img width="373" alt="image" src="https://github.com/user-attachments/assets/43f32aee-c043-409b-bb2a-ba713c9976b0" />
<img width="371" alt="image" src="https://github.com/user-attachments/assets/7c83571c-2299-48eb-9ecd-b1c2cce5bc52" />
</p>

<p align="center">
<img width="366" alt="image" src="https://github.com/user-attachments/assets/f074717c-a575-4d9b-957d-6e075af4269e" />
<img width="362" alt="image" src="https://github.com/user-attachments/assets/944ead3e-3884-4a69-aa66-675ea9d5f63a" />
</p>

## Comparison: YOLO v1 vs. YOLO v7
### 1. Architecture Design
- **YOLO v1:** Uses a single convolutional neural network to predict bounding boxes and class probabilities directly.
- **YOLO v7:** Introduces features like **Extended Efficient Layer Aggregation Networks (E-ELAN)** and **dynamic label assignment**, improving detection accuracy.

### 2. Loss Function
- **YOLO v1:** Uses sum-squared error loss, treating localization and classification errors equally.
- **YOLO v7:** Uses **multiple loss terms** (CIoU loss for bounding boxes, BCE loss for objectness, and classification loss), improving convergence and accuracy.

### 3. Computational Complexity
- **YOLO v1:** Designed for real-time detection but struggles with small objects and overlapping detections.
- **YOLO v7:** More efficient model design using **re-parameterized convolutions** and **bag of freebies** optimizations, making it faster and more accurate.

### Summary
YOLO v7 addresses YOLO v1’s limitations by improving feature aggregation, refining loss functions, and optimizing computations for better accuracy and speed.

## References
- [YOLO v7 Paper](https://arxiv.org/abs/2207.02696)
- [YOLO v1 Paper](https://arxiv.org/abs/1506.02640)
