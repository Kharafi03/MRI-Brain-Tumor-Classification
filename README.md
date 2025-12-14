# README: MRI Brain Tumor Classification

## Overview
This project focuses on classifying brain tumors from MRI images using a deep learning approach, specifically leveraging Transfer Learning with the ResNet50V2 model. The goal is to accurately distinguish between different types of brain tumors (glioma, meningioma, pituitary) and healthy brains (notumor). The project explores the impact of various optimizers (Adam, RMSprop, Nadam, Adamax) on model performance.

## Dataset
The dataset used is the "Brain Tumor MRI Dataset" available on Kaggle (masoudnickparvar/brain-tumor-mri-dataset). It contains MRI images categorized into four classes:
- Glioma
- Meningioma
- Notumor (healthy brain)
- Pituitary

## Preprocessing
To prepare the MRI images for model training, the following preprocessing steps are applied:
1.  **Image Loading**: MRI images are loaded using OpenCV.
2.  **Thresholding**: Images are binarized to highlight brain regions.
3.  **Erosion & Dilation**: Morphological operations are applied to refine the binary mask.
4.  **Contour Detection & Cropping**: The largest contour (assumed to be the brain) is detected, and the image is cropped to focus on this region.
5.  **Resizing**: Cropped images are resized to a target dimension of (224, 224) pixels.
6.  **Normalization**: Pixel values are normalized to the range [0, 1].

Additionally, **data augmentation** techniques such as rotation and horizontal flipping are used during training to increase the diversity of the training data and improve model generalization.

## Model Training

### Transfer Learning with ResNet50V2
The project utilizes ResNet50V2, a pre-trained Convolutional Neural Network (CNN) on the ImageNet dataset, as a base model for transfer learning. The top layers of the ResNet50V2 are fine-tuned, while the initial layers are frozen to retain the learned features.

### Model Architecture
The architecture consists of:
-   **Base Model**: ResNet50V2 (pre-trained on ImageNet, `include_top=False`)
-   **Fine-tuning**: The top 20% of ResNet50V2 layers are unfrozen and trained.
-   **Classification Head**: Added on top of the base model:
    -   `GlobalAveragePooling2D` layer
    -   `Dense` layer with 64 units, `relu` activation, and `l2` regularization
    -   `Dropout` layer (0.3 rate)
    -   `Dense` output layer with 4 units (for 4 classes) and `softmax` activation.

### Optimizers Explored
The model was trained with four different optimizers, each with a learning rate of `0.00005`, over 15 epochs:
-   **Adam**
-   **RMSprop**
-   **Nadam**
-   **Adamax**

**Callbacks**: `ModelCheckpoint` is used to save the best performing model based on validation loss.

## Results

All optimizers demonstrated excellent performance on the classification task. The following table summarizes the key metrics for each optimizer on the test set:

| Optimizer | Accuracy | Precision | Recall | F1-Score |
| :-------- | :------- | :-------- | :----- | :------- |
| Adam      | 0.9916   | 0.9916    | 0.9916 | 0.9916   |
| RMSprop   | 0.9908   | 0.9909    | 0.9908 | 0.9909   |
| Nadam     | 0.9893   | 0.9894    | 0.9893 | 0.9893   |
| Adamax    | 0.9863   | 0.9865    | 0.9863 | 0.9863   |

The **Adam optimizer** achieved the highest overall performance with an accuracy, precision, recall, and F1-score of **0.9916**.

Detailed classification reports and confusion matrices for each optimizer are provided in the notebook, showing strong performance across all individual classes as well.

## Conclusion
The fine-tuned ResNet50V2 model with data augmentation proved highly effective in classifying brain tumors from MRI images. The Adam optimizer slightly outperformed the others, achieving near-perfect accuracy and F1-scores on the test set. This project demonstrates a robust approach for medical image classification using transfer learning.