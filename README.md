# CSCI 653 Project: Training Acceleration of 3D Generative AI

Name: Zhangyu Jin

USC Student ID: 3350-2383-72

## (a) Project Description

This project aims to optimize 3D generative AI models to reduce their significant training demands on GPUs. 
Specifically, we're developing an architecture that generates a 3D asset from a single 2D image. 
Our approach focuses on several techniques to lower both CUDA memory usage and training time. 
We'll implement model-level optimizations such as Flash Attention, Sparse Convolutions, and Elastic Memory Management.

<img src="file:///C:/Users/zjin/Downloads/codes/CSCI653_Project/assets/output.gif" title="" alt="Demo of Feature" data-align="center">

## (b) Model Architecture

SS-Flow Transformer has 0.5 Billion parameters, and SLat-Flow Transformer also has 0.5 Billion parameters. The network is so huge, thus we need special techniques to accelerate training speed and reduce training memory cost.

![](C:\Users\zjin\Downloads\codes\CSCI653_Project\assets\Snipaste_2025-12-01_02-42-33.png)

## (c) Acceleration Methods

- FlashAttention: The standard attention mechanism in transformers is a major bottleneck, consuming vast amounts of CUDA memory and computation time. This limitation often prevents the use of larger, more powerful models. FlashAttention is a more memory-efficient and faster algorithm that overcomes these issues by rethinking how attention is calculated, significantly reducing both memory and time costs.

- Sparse Convolution: 3D generative models often rely on 3D data representations like voxels. A typical resolution, such as 64×64×64, results in over 262,000 data points, making it computationally expensive for traditional convolution and transformer layers. Sparse convolution is a memory-efficient technique that intelligently processes this 3D data. It focuses computation only on the non-empty parts of the voxel grid, drastically cutting down on unnecessary calculations and memory usage.

- Elastic Memory Management: This technique involves dynamically allocating and deallocating memory during the training process. Instead of reserving a fixed, large block of memory from the start, elastic memory management adjusts memory usage on the fly based on the model's current needs. This prevents memory waste and allows larger models to be trained on hardware with limited resources by making smarter use of available memory.

## (d) Expected Results

- Reduced CUDA Memory Usage: We aims to see a 20-30% decrease in peak memory usage, allowing us to train larger models and use bigger batch sizes without running out of memory. 

- Faster Training Time: We want to see a training time reduction of 25%.
