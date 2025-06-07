# Classification and Visualization of Plant Leaf Diseases Using Deep Learning and Image Processing Techniques

This repository contains the code, datasets, and documentation for the research project titled **"Classification and Visualization of Plant Leaf Diseases Using Deep Learning and Image Processing Techniques"** conducted by students from the Department of Computer Science and Engineering at the Islamic University of Technology, Dhaka, Bangladesh. Additionally, it includes a Streamlit-based application called **PlantCare**, designed for farmers, researchers, and general users to detect plant diseases and learn about plant health.

---

## Project Overview

Plant diseases are a major threat to global agriculture, causing significant crop losses and economic damage. Early and accurate detection is critical for effective disease management. This project evaluates various deep learning models for automated plant disease classification using two datasets: the **New Plant Diseases Dataset** (87,000 images, 38 classes) and the **Plant Village Dataset** (70,000 images, 9 plant species). We tested models like CNN, AlexNet, ResNet, EfficientNet, MobileNetV2, DenseNet121, VGG16, InceptionV3, and VanillaCNN, achieving high accuracy and integrating Explainable AI (XAI) techniques like Grad-CAM for interpretability.

We also developed **PlantCare**, a user-friendly web application built with Streamlit, allowing users to upload plant leaf images and detect diseases in real-time.

---

## Authors

- **Namisa Najah**
- **Nafisa Binte Ghulam Kibria**
- **Faiza Maliat**
- **Ishmaam Iftekhar Khan**
- **Nusrat Siddique**

**Affiliation**: Department of Computer Science and Engineering, Islamic University of Technology, Dhaka, Bangladesh

---

## Abstract

The early identification of plant diseases is vital for preventing crop damage and ensuring food security. This study leverages deep learning models for automated plant disease detection, focusing on their efficiency in image classification tasks. We evaluated multiple architectures—CNN, AlexNet, ResNet, EfficientNet, MobileNetV2, DenseNet121, VGG16, InceptionV3, and VanillaCNN—on two datasets: the New Plant Diseases Dataset (87,000 images, 38 classes) and the Plant Village Dataset (70,000 images, 9 species). Key results include:

- **New Plant Diseases Dataset**: CNN (95.94%), ResNet (99.07%), MobileNetV2 (92.12%).
- **Plant Village Dataset**: EfficientNet (99.40%), AlexNet (96.71%), ResNet50 (96.19%).

EfficientNet outperformed other models, while InceptionV3 (18.07%) and DenseNet121 (47%) showed lower performance. These findings contribute to developing efficient, interpretable plant disease detection systems to support farmers and improve agricultural productivity.

---

## Methodology

### Datasets

1. **New Plant Diseases Dataset**:
   - 87,000 images
   - 38 classes (14 crop species, healthy and diseased states)
2. **Plant Village Dataset**:
   - 70,000 images
   - 9 plant species (high-quality images of healthy and diseased leaves)

### Image Pre-Processing

- **Loading**: Images loaded with inferred labels.
- **Resizing**: Uniformly resized to 128x128 pixels.
- **Normalization**: Pixel values scaled to [0,1].
- **Shuffling**: Randomized to reduce bias.
- **Data Augmentation**: Applied random rotation, flipping, and zooming for generalization.

### Models Evaluated

- **CNN**: Custom architecture with convolutional and fully connected layers.
- **AlexNet**: 5 convolutional layers, 3 fully connected layers, ReLU, and dropout.
- **EfficientNet**: Optimized with compound scaling for efficiency and accuracy.
- **ResNet**: Skip connections to mitigate vanishing gradient issues.
- **MobileNetV2**: Lightweight with depthwise separable convolutions.
- **DenseNet121**: Dense connectivity for feature propagation.
- **VGG16**: Uniform deep CNN for hierarchical feature extraction.
- **InceptionV3**: Factorized convolutions for multi-scale feature capture.
- **VanillaCNN**: Baseline CNN trained from scratch.

### Explainable AI (XAI)

- **Grad-CAM**: Visualizes model focus areas with heatmaps (red: high attention, blue: low attention).

---

- **Key Insights**:
  - ResNet excels on the New Plant Diseases Dataset (99.07%).
  - EfficientNet leads on the Plant Village Dataset (99.40%).
  - InceptionV3 and DenseNet121 underperform on the Plant Village Dataset.

---

## Plant Care Application

### Overview

**Plant Care** is a Streamlit-based web application designed for:

- **Farmers**: Detect plant diseases in real-time.
- **General Users**: Learn about plant health.
- **Researchers**: Analyze leaf images for study purposes(multiple datasets)

Users can upload leaf images, and the app uses our trained deep learning model to classify diseases and provide insights.

### Features

- Upload plant leaf images.
- Real-time disease detection.
- Visualizations (Grad-CAM heatmaps) for interpretability.
- User-friendly interface.

### Tech Stack

- **Framework**: Streamlit
- **Backend**: Python, TensorFlow/Keras (for model inference)
- **Models**: Pre-trained deep learning models

---

## Installation

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (optional but recommended)

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/FaizaM07/Machine-Learning-Project.git
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv <venv_name>
   <venv_name>\Scripts\activate
   ```

   `OR`

   use conda virtual environment

   ```bash
   conda create --name <env_name>
   conda activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Get the models**:

   Run all the notebooks; download the `keras` files and add them in the <PlantCare> folder. Two models are added already.

5. **Get Started**:
   ```bash
   streamlit run Plant.py
   ```
