About the Project

Alzheimer's disease is a progressive neurological disorder that affects memory and cognitive abilities. Detecting it at an early stage can help in timely treatment and better patient care. The objective of this project is to classify brain MRI scans into different stages using a deep learning approach based on Vision Transformers (ViT).

I built this project to understand the complete workflow of a computer vision application—from preprocessing MRI images and training a deep learning model to generating explainable predictions and serving the model through a REST API. Along with image classification, I also integrated Grad-CAM to visualize which regions of the MRI contributed the most to the model's prediction, making the results more interpretable.

Dataset

The model is trained on a dataset of T1-weighted brain MRI scans, a commonly used imaging modality for Alzheimer's disease analysis because it captures detailed structural information about the brain. Public Alzheimer's MRI datasets are commonly derived from sources such as the Alzheimer's Disease Neuroimaging Initiative (ADNI) and similar research repositories.

For this project, the images are organized into three categories:

AD (Alzheimer's Disease) – Patients diagnosed with Alzheimer's disease.
CI (Cognitively Impaired) – Patients showing cognitive impairment or mild cognitive decline.
CN (Cognitively Normal) – Healthy individuals without signs of cognitive impairment.

The dataset is divided into separate training and testing folders, making it easy to train the model and evaluate its performance on unseen images.

 Data Preprocessing

Before training the model, each MRI image goes through several preprocessing steps to ensure consistent input:

Images are resized to 224 × 224 pixels to match the input size expected by the Vision Transformer.
Pixel values are normalized using ImageNet mean and standard deviation since the model is initialized with pretrained ImageNet weights.
Data augmentation techniques such as random transformations are applied to the training images to improve the model's ability to generalize.
Separate preprocessing pipelines are used for training and testing to ensure unbiased evaluation.

These preprocessing steps help the model learn more robust features while reducing overfitting.

 Model

The project uses Vision Transformer (ViT-Base Patch16-224), a transformer-based architecture that treats an image as a sequence of small patches instead of relying on convolutional filters like traditional CNNs.

Instead of extracting only local features, the self-attention mechanism allows the model to capture relationships between different regions of the MRI image. This makes Vision Transformers particularly effective for complex image classification tasks when combined with transfer learning.

The pretrained ViT model is fine-tuned for a three-class Alzheimer's classification problem.

 Technologies Used

This project was developed using the following technologies and libraries:

Python
PyTorch
Vision Transformer (ViT)
timm
Torchvision
NumPy
OpenCV
Pillow (PIL)
Scikit-learn
Matplotlib
FastAPI
Uvicorn
Grad-CAM
Model Evaluation

The trained model is evaluated using multiple performance metrics to provide a complete understanding of its classification performance:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report

The project also saves the best-performing model during training and generates training curves for loss and evaluation metrics.

 Explainability

To improve transparency, I integrated Grad-CAM into the project. Instead of only predicting a class label, Grad-CAM generates a heatmap that highlights the regions of the MRI scan that had the greatest influence on the model's prediction. This makes it easier to understand how the model arrives at its decisions and provides an additional level of interpretability for medical image classification.

Future Improvements

There are several features that can be added in the future to make the project more practical and production-ready:

Web-based interface for uploading MRI images
Patient report generation
Docker containerization
Cloud deployment (AWS/Azure)
User authentication
Prediction history and database integration
Model optimization for faster inference
Support for additional Alzheimer's severity stages
