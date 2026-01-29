# Bird Species Classification using CNN (MobileNetV2)

This project identifies Indian bird species from images using a Convolutional Neural Network (CNN) with transfer learning (MobileNetV2).

## Dataset

Dataset used: **25 Indian Bird species with 22.6k images** (Kaggle)

Kaggle link:  
https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images

> Note: The dataset is NOT included in this repository to keep the repo light.  
> To run this project, download the dataset from Kaggle and place it under `data/` as described in the code.

## Model

- Base model: MobileNetV2 (pretrained on ImageNet)
- Custom classification head with:
  - GlobalAveragePooling2D
  - Dropout
  - Dense layer with 25 outputs (softmax)

## How to Run

1. Create virtual environment and install requirements
2. Download dataset from Kaggle
3. Run `CNN_train.py` to train
4. Run `evaluate_model.py` to see accuracy and confusion matrix
5. Use `predict_single.py` to predict a bird from an image
