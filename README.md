# MRI Brain Tumor Classification

## Overview
This project aims to classify MRI images of the brain to detect the presence and type of brain tumors. It utilizes a deep learning approach with Transfer Learning, specifically fine-tuning a ResNet50V2 model, and evaluates different optimizers to find the best performing one.

## Dataset
The dataset used for this project is the "Brain Tumor MRI Dataset" from Kaggle. It contains MRI images categorized into four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary

The dataset is split into `Training` and `Testing` subsets.

## Preprocessing
Image preprocessing is a crucial step to enhance the quality of the images and prepare them for model training. The `preprocess_image` function performs the following steps:
1.  **Read Image**: Images are read in color mode.
2.  **Grayscale Conversion**: Converted to grayscale to simplify the image.
3.  **Gaussian Blur**: Applied to reduce noise.
4.  **Thresholding**: Binary thresholding is used to segment the image and isolate the brain region.
5.  **Contour Detection**: The largest contour is found, which is assumed to be the brain area.
6.  **Cropping**: The image is cropped to focus on the brain region.
7.  **Resizing**: Images are resized to `(224, 224)` pixels, which is the input size expected by ResNet50V2.
8.  **Normalization**: Pixel values are scaled to the range `[0, 1]`.

Labels are one-hot encoded using `LabelBinarizer`.

## Data Augmentation
To increase the diversity of the training data and improve the model's generalization capabilities, data augmentation is applied using `ImageDataGenerator` with the following transformations:
-   `rotation_range=10`: Randomly rotates images by up to 10 degrees.
-   `horizontal_flip=True`: Randomly flips images horizontally.
-   `fill_mode='nearest'`: Fills in newly created pixels after rotation or shifting with the nearest available pixel.

## Model Architecture (ResNet50V2 Transfer Learning)
The project utilizes a pre-trained ResNet50V2 model as a base for transfer learning. The model architecture is as follows:
1.  **Base Model**: ResNet50V2 is loaded with `include_top=False` (to remove the classification head) and `weights="imagenet"` (for pre-trained weights).
2.  **Fine-Tuning**: The first 80% of the base model's layers are frozen, and the remaining 20% are unfrozen to allow for fine-tuning on the new dataset.
3.  **Custom Classification Head**: A custom head is added on top of the base model:
    -   `GlobalAveragePooling2D`: Reduces spatial dimensions.
    -   `Dense(64, activation='relu', kernel_regularizer=l2(0.001))`: A dense layer with ReLU activation and L2 regularization to prevent overfitting.
    -   `Dropout(0.3)`: Dropout layer to further prevent overfitting.
    -   `Dense(4, activation='softmax')`: Output layer with 4 units (for 4 classes) and softmax activation.

## Optimizers Evaluated
The model's performance is evaluated using different optimizers, each with a learning rate of `0.00005`:
-   **Adam**
-   **RMSprop**
-   **Nadam**
-   **Adamax**

## Training and Evaluation
Each model is trained for 15 epochs with a `batch_size` of 32. `ModelCheckpoint` is used to save the best model weights based on validation loss. Performance metrics include accuracy, precision, recall, and F1-score.

## Results
The performance metrics (Accuracy, Precision, Recall, F1-Score) and Confusion Matrices for each optimizer are generated and compared. Additionally, training vs. validation accuracy and loss curves are plotted to visualize the training progress and identify potential overfitting.

### Summary of Performance Metrics per Optimizer

| Optimizer | Accuracy | Precision | Recall | F1-Score |
|:----------|:---------|:----------|:-------|:---------|
| Adam      | 0.9939   | 0.9939    | 0.9939 | 0.9939   |
| RMSprop   | 0.9908   | 0.9909    | 0.9908 | 0.9908   |
| Nadam     | 0.9886   | 0.9886    | 0.9886 | 0.9886   |
| Adamax    | 0.9817   | 0.9819    | 0.9817 | 0.9816   |


### Key Findings:
-   **Adam Optimizer** generally shows strong performance, achieving high accuracy, precision, recall, and F1-score.
-   **RMSprop** and **Nadam** also demonstrate competitive results.
-   **Adamax** tends to have slightly lower performance compared to the other optimizers.

The plots for Accuracy, Precision, Recall, and F1-Score comparison provide a visual summary of which optimizer performs best across different metrics.

## Conclusion
Based on the evaluation, the **Adam** optimizer seems to provide the most consistent and highest performance for this brain tumor classification task, closely followed by RMSprop and Nadam. The fine-tuned ResNet50V2 model, coupled with appropriate preprocessing and data augmentation, achieves high classification accuracy.