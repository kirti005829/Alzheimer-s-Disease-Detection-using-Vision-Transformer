# Alzheimers-Detection-ViT
## Introduction
The notebook `ViT_Based_Alzheimer_Classifier.ipynb` provides a step-by-step guide to classifying brain MRI images into different categories related to Alzheimer's Disease. The core of the solution is a Vision Transformer (ViT) model, pre-trained on the ImageNet-1K dataset, which is then fine-tuned for this specific classification task. This approach leverages the powerful feature extraction capabilities of large-scale pre-trained models.

***

## Dataset
The project uses a custom dataset of brain MRI images. The dataset is structured into three classes:
- **AD**: Alzheimer's Disease
- **CI**: Cognitive Impairment
- **CN**: Cognitively Normal
## Model Architecture
The classifier is based on the Vision Transformer (ViT) architecture. Specifically, it uses `vit_b_16` from `torchvision.models`.

The pre-trained ViT model, originally trained on ImageNet-1K, is adapted for this 3-class classification problem by replacing the final classification head.
- **Backbone**: `ViT_B_16_Weights.IMAGENET1K_V1`
- **New Head**: A custom `nn.Linear` layer with an output size of 3, corresponding to the three classes (AD, CI, CN).

The model is trained using:
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizer**: AdamW (`optim.AdamW`)
- **Learning Rate**: $1e-4$
- **Weight Decay**: 0.01
- **Epochs**: 5

***## Dependencies
The following libraries are required to run the notebook:
- `zipfile`
- `os`
- `matplotlib`
- `numpy`
- `torch`
- `torchvision`
- `torch.nn`
- `torch.optim`
- `seaborn`
- `pandas`
- `PIL`

The notebook assumes a GPU is available and will use it if `torch.cuda.is_available()` returns `True`.
