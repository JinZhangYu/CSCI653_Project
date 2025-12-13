# CSCI 653 Project: Training Acceleration of 3D Generative AI

Name: Zhangyu Jin

USC Student ID: 3350-2383-72

## (A) Project Description

This project aims to optimize 3D generative AI models to reduce their significant training demands on GPUs. 
Specifically, we're developing an architecture that generates a 3D asset from a single 2D image. 
Our approach focuses on several techniques to lower both CUDA memory usage and training time. 
We'll implement model-level optimizations such as Flash Attention, Sparse Convolutions, and Elastic Memory Management.

## (B) What is 3D Generative AI

3D Generative AI refers to a branch of artificial intelligence focused on creating three-dimensional models, scenes, or environments from various input modalities such as text, 2D images, or sparse geometric data. By leveraging these 2D visual cues and basic structural data, the AI can infer and reconstruct complex 3D architectures, effectively turning flat, overhead views into fully realized, volumetric cityscapes.

![Demo of Feature](assets/output_small.gif)

## (C) Current Problems

SS-Flow Transformer has 0.5 Billion parameters, and SLat-Flow Transformer also has 0.5 Billion parameters. The network is so huge, thus we need special techniques to accelerate training speed and reduce training memory cost.

<img title="" src="file:///C:/Users/zjin/Downloads/codes/CSCI653_Project/assets/Snipaste_2025-12-12_23-12-49.png" alt="" width="503" data-align="center">

## (D) Methods

### (D.1) Flash-Attention

The standard attention mechanism in transformers is a major bottleneck, consuming vast amounts of CUDA memory and computation time. This limitation often prevents the use of larger, more powerful models. FlashAttention is a more memory-efficient and faster algorithm that overcomes these issues by rethinking how attention is calculated, significantly reducing both memory and time costs.

![FlashAttention](C:\Users\zjin\Downloads\codes\CSCI653_Project\assets\flashattn_banner.jpg)

### (D.2) Sparse Convolution

3D generative models often rely on 3D data representations like voxels. A typical resolution, such as 64×64×64, results in over 262,000 data points, making it computationally expensive for traditional convolution and transformer layers. Sparse convolution is a memory-efficient technique that intelligently processes this 3D data. It focuses computation only on the non-empty parts of the voxel grid, drastically cutting down on unnecessary calculations and memory usage.

<img title="" src="file:///C:/Users/zjin/Downloads/codes/CSCI653_Project/assets/nrupa_conv_exp.png" alt="" width="526" data-align="center">

### (D.3) Activation Checkpointing

This technique involves dynamically allocating and deallocating memory during the training process. Instead of reserving a fixed, large block of memory from the start, elastic memory management adjusts memory usage on the fly based on the model's current needs. This prevents memory waste and allows larger models to be trained on hardware with limited resources by making smarter use of available memory.

![](C:\Users\zjin\Downloads\codes\CSCI653_Project\assets\Snipaste_2025-12-12_23-37-23.png)

## (E) Results

All of the experiments are conducted on 4xA100 and 40K training iterations.

### (E.1) Flash Attention

If we use normal attention, we cannot fit the network inside A100. That means we would get cuda out of memory error. So after we apply the Flash-Attention 2, the training time is roughly 26 hours.

| Attention Type    | Time Cost |
| ----------------- | --------- |
| Normal Attention  | OOM       |
| Flash-Attention 2 | 26 h      |

### (E.2) Sparse Convolution

If we use normal 3D convolution, then we will also face with the memory issue during training time. As mentioned in the last section, we apply sparse 3D convolution at the beginning of SLat VAE encoder and also the SLat flow transformer. The OOM issue is resolved.

| Convolution Type | Time Cost |
| ---------------- | --------- |
| Normal 3D Conv   | OOM       |
| Sparse 3D Conv   | 26 h      |
