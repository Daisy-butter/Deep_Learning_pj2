# Neural Network Explorations on CIFAR-10 🚀

Welcome to the CIFAR-10 playground! This repository contains experiments to help you understand how different neural network components affect performance. Perfect for when you want to see the magic of deep learning in action!

## 🛠️ Experiment Environment
All experiments were conducted on a single-GPU system with the following specs:
- **CUDA Version:** 12.9
- **CPU:** Intel Core i9-13900HX
- **GPU:** NVIDIA GeForce RTX 4060
- **Training Protocol:** 100 epochs with batch size of 128

## 🎯 Experiments

### 1. Neural Network Playground (`basecnn.py`) 
Time to mix and match! This script lets you experiment with:
- Different activation functions (ReLU, Sigmoid, Tanh - oh my!)
- Various loss functions
- Filter combinations in your CNN

**Cool outputs:** 
- Total runtime and best accuracy printed to console
- Results automatically saved in structured format
- Want to visualize kernels? Run `visualize_kernel.py` for a peek inside your CNN's brain!

### 2. Optimizer Olympics (`basecnn_optimize.py`) 
Watch SGD, Adam, and Adam+StepLR compete for the training championship! This script:
- Compares how different optimizers affect training
- Saves visual results so you can see who's winning

### 3. Batch Normalization Showdown (`bn_experiments.py`) 
The great BN debate - to normalize or not to normalize? This experiment:
- Trains VGG models with and without BatchNorm
- Creates `bn_comparison.png` showing loss/accuracy differences
- Generates loss landscapes (`loss_landscape_bn.png` vs `loss_landscape_no_bn.png`) - because who doesn't love a good landscape?

## 💡 About Batch Normalization
BatchNorm is like the yoga instructor for your neural network - it helps keep everything flexible and well-balanced during training. By normalizing layer inputs, it:
- Makes training faster and more stable
- Reduces sensitivity to initialization
- Acts as a mild regularizer
But is it always better? Run the experiment to find out!

## 🏃‍♀️ Getting Started
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the experiments you're interested in
4. Check the generated results and visualizations

Happy experimenting! May your gradients flow smoothly and your accuracy be high! 🎯
