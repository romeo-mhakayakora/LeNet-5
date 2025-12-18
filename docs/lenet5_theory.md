# LeNet-5: Theoretical Overview

LeNet-5 is a classic convolutional neural network (CNN) architecture originally designed for digit recognition. Its core idea is to learn **local features** with convolution, progressively build **hierarchical representations**, and then map those features to classes with fully-connected layers.

## Why CNNs Work (intuition)

- **Locality**: nearby pixels are correlated; small filters (kernels) can detect edges/corners/strokes.
- **Weight sharing**: the same filter slides across the image, so the model learns a feature once and can detect it anywhere (translation equivariance).
- **Downsampling**: pooling reduces spatial resolution, making representations more robust to small shifts/noise and lowering compute.

## Architecture (as implemented in this repo)

This project pads MNIST from **2828** to **3232** (the original LeNet-5 input size) and uses a LeNet-style stack:

```text
Input: 32321
  C1: Conv 55, 6 filters, tanh  -> 32326   (padding=same)
  S2: AvgPool 22, stride 2      -> 16166
  C3: Conv 55, 16 filters, tanh -> 161616 (padding=same)
  S4: AvgPool 22, stride 2      ->  8816
  C5: Conv 55, 120 filters, tanh->  88120 (padding=same in this code)
  F6: Dense 84, tanh
  F7: Dense 10, softmax
```

> Note: The *original* LeNet-5 used some `valid` convolutions and a slightly different C5 shape (often 11120 after valid convs). This repo keeps `padding="same"` for simplicity and stable shapes in modern frameworks.

## Layer-by-layer: what each part learns

### Convolution layers (C1/C3/C5)

A 2D convolution applies a small kernel over the image to produce feature maps. Each filter learns a pattern (e.g., vertical edge, curve, stroke junction).

- Early layers (C1) learn **simple** patterns (edges, small blobs).
- Middle layers (C3) learn **combinations** of edges (corners, curves).
- Deeper layers (C5) learn **digit parts** and more global structure.

### Average pooling (S2/S4)

Average pooling downsamples the feature maps by averaging local neighborhoods. It reduces spatial resolution and makes features less sensitive to small translations. Classic LeNet used **average pooling** (modern CNNs often use max pooling).

### Activation: `tanh`

`tanh` squashes values to (-1, 1). This is historically consistent with LeNet-5. (Many modern CNNs prefer ReLU for faster training, but `tanh` is fine for a faithful LeNet-style model.)

### Fully connected layers (F6/F7)

After flattening, dense layers combine all learned features to produce class probabilities.

- **F6 (84 units)** acts like a learned feature combiner.
- **F7 (softmax)** outputs a probability distribution over the 10 digit classes.

## Training objective

- The model outputs $\hat{y}$ via softmax.
- We minimize **categorical cross-entropy**:

$$\mathcal{L}(y, \hat{y}) = -\sum_{k=1}^{10} y_k \log(\hat{y}_k)$$

## Optimization and learning-rate schedule (in this repo)

- Optimizer: **SGD** (stochastic gradient descent).
- Schedule: keep the initial learning rate for the first 10 epochs, then multiply by 0.1 (a simple step decay).
- Checkpointing: the best model is saved based on **validation accuracy**.

## Practical notes for MNIST

- Padding to 3232 aligns with the original LeNet-5 input size.
- Normalizing pixel values to [0, 1] improves optimization stability.
- One-hot labels are required for categorical cross-entropy with a softmax output.
