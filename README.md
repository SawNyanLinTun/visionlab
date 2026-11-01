# Image Classification Using CNN — Google Colab

**A Computer Vision Project Built from Scratch with Python and PyTorch**

## Project Overview

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using Google Colab.

The objective is to develop a model capable of recognizing three geometric shapes: **circles, squares, and triangles**.

Instead of using a pretrained model or downloading an existing dataset, the project generates synthetic images programmatically and trains a custom CNN from scratch.

The project covers the complete machine learning workflow, from dataset preparation to model evaluation and exporting the trained model.

## Project Objectives

- Understand how digital images are represented as numerical arrays.
- Generate a custom labeled image dataset using Python.
- Prepare training, validation, and testing datasets.
- Design a CNN architecture using PyTorch.
- Train the network using backward propagation and gradient-based optimization.
- Analyze training and validation performance.
- Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.
- Save and reload a trained model for future inference.

## Technologies Used

| Technology | Purpose |
|---|---|
| Google Colab | Cloud-based development environment |
| Python | Main programming language |
| PyTorch | Neural network development and training |
| NumPy | Numerical and image array processing |
| Pillow (PIL) | Synthetic image generation |
| Matplotlib | Image and metric visualization |
| pandas | Training history management |
| scikit-learn | Classification performance evaluation |

## 1. Dataset Generation

The project uses a custom synthetic image dataset generated with Python's Pillow library.

Three image classes are included:

- Circle
- Square
- Triangle

Each image has a resolution of **64 × 64 pixels** with three RGB colour channels.

To introduce variation, the image generator randomizes:

- Object colour
- Background colour
- Object size
- Object position

This helps reduce reliance on fixed colours or positions when learning to classify shapes.

### Dataset Distribution

A total of **900 images** were generated, containing 300 images per class.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 630 | 70% |
| Validation | 135 | 15% |
| Testing | 135 | 15% |
| **Total** | **900** | **100%** |

The training dataset is used to update model parameters, the validation dataset is used to monitor performance and select the best checkpoint, and the test dataset is reserved for final evaluation.

## 2. CNN Architecture

The model was developed from scratch using PyTorch's `nn.Module`.

### Model Structure

```text
Input Image (3 × 64 × 64)
          |
          v
Convolutional Layer
3 input channels → 16 feature maps
          |
         ReLU
          |
    Max Pooling (2×2)
          |
          v
Convolutional Layer
16 input channels → 32 feature maps
          |
         ReLU
          |
    Max Pooling (2×2)
          |
          v
        Flatten
          |
Fully Connected Layer
       64 neurons
          |
         ReLU
          |
   Output Layer
      3 classes
```

The convolutional layers learn visual features, while the fully connected layers use those features to produce classification scores.

The model predicts one of three labels: `circle`, `square`, or `triangle`.

## 3. Model Training

The CNN was trained using the custom synthetic image dataset.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Model | Custom ShapeCNN |
| Framework | PyTorch |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Epochs | 15 |
| Batch size | 32 |
| Input resolution | 64 × 64 |
| Output classes | 3 |

During each training epoch, the model performs forward propagation, calculates the classification loss, applies backpropagation, and updates its parameters.

Validation performance is measured after each epoch.

The checkpoint with the highest validation accuracy is retained for final evaluation.

## 4. Training Performance Analysis

Training and validation results are recorded throughout the experiment.

Two learning curves are generated:

**Accuracy Curve**

Shows how classification accuracy changes as the CNN learns.

**Loss Curve**

Shows how prediction error changes during training.

Comparing training and validation curves helps identify potential overfitting and understand whether the model is improving on unseen validation examples.

The training history is saved as:

`models/m4_training_history.csv`

## 5. Model Evaluation

After training, the selected CNN checkpoint is evaluated using the held-out test dataset.

### Evaluation Metrics

**Accuracy:** Measures the proportion of correctly classified test images.

**Precision:** Measures the proportion of predictions for a particular class that are correct.

**Recall:** Measures the proportion of actual images belonging to a class that are correctly identified.

**F1-score:** Combines precision and recall into one measurement.

**Confusion Matrix:** Shows correct classifications and errors across the three image categories.

The experiment also includes visual inspection of individual predictions and misclassified images to support error analysis.

### Experimental Results

The notebook calculates and displays:

- Final test accuracy
- Average test loss
- Per-class precision, recall, and F1-score
- Confusion matrix
- Correct and incorrect prediction examples

Exact performance figures are recorded in the notebook output and `m4_test_metrics.json`; no unverified accuracy figure is reported here.

## 6. Testing the Saved Model

After training and evaluation, the saved checkpoint is loaded into a fresh instance of the same CNN architecture.

The model is then tested on newly generated shape images.

The prediction process includes:

1. Generate a new synthetic image.
2. Convert it into a normalized RGB tensor.
3. Pass the tensor through the trained CNN.
4. Apply Softmax to obtain class probabilities.
5. Display the predicted label and confidence scores.

A prediction gallery is also generated to compare actual and predicted classes across multiple samples.

## 7. Project Outputs

The Colab experiment produces the following files:

```text
M4_Computer_Vision_CNN_From_Scratch.ipynb

models/
├── m4_shapes_cnn.pt
├── m4_training_history.csv
└── m4_test_metrics.json
```

### File Descriptions

| File | Description |
|---|---|
| `M4_Computer_Vision_CNN_From_Scratch.ipynb` | Complete Colab experiment |
| `m4_shapes_cnn.pt` | Trained CNN checkpoint |
| `m4_training_history.csv` | Training and validation metrics |
| `m4_test_metrics.json` | Test metrics and confusion matrix |

The model checkpoint contains learned parameters and metadata such as class labels and architecture name.

## 8. Running the Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `M4_Computer_Vision_CNN_From_Scratch.ipynb`.
3. Select a Python runtime. CPU is sufficient for this experiment.
4. Execute the notebook cells sequentially.
5. Inspect the generated datasets, training curves, evaluation metrics, and predictions.
6. Download the checkpoint and results after completing the experiment.

The notebook generates the required synthetic images, so an external dataset is not necessary.

## 9. Key Learning Outcomes

Through this project, I developed practical experience in:

- Creating a labeled computer vision dataset from scratch.
- Working with image tensors and RGB pixel representations.
- Implementing a convolutional neural network using PyTorch.
- Understanding training loops, loss functions, optimizers, and backpropagation.
- Separating training, validation, and testing workflows.
- Evaluating classification models with multiple performance metrics.
- Investigating incorrect predictions.
- Saving and reloading neural network checkpoints.

## 10. Limitations

This project is an educational computer vision experiment.

The dataset contains simple synthetic geometric shapes rather than real-world photographs.

Although randomization introduces variation, the generated images remain relatively simple and may not capture complex lighting, textures, backgrounds, or object transformations.

Consequently, strong performance on this dataset does not establish that the model will generalize to real-world image classification tasks.

*Built from scratch in Google Colab for computer vision learning*
