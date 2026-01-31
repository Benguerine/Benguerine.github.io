---
title: "Making Deep Learning Faster: Quantization Benchmarks with TensorRT"
date: 2026-01-31
categories: [Deep Learning, Inference, Optimization]
tags: [TensorRT, Quantization, FP32, FP16, INT8, ONNX]
---

## Introduction

Modern deep learning models are typically trained using **FP32 (32-bit floating-point precision)** to ensure numerical stability and maximum accuracy. However, when deploying these models in production, **FP32 inference is often inefficient**, leading to higher **latency**, increased **memory consumption**, and lower **throughput**.

To address these limitations, **quantization** is widely used to reduce numerical precision during inference. In this article, we analyze and benchmark **FP32**, **FP16**, and **INT8** inference using **NVIDIA TensorRT**, with a focus on performance trade-offs and deployment considerations.

---

## Quantization in Deep Learning Inference

**Quantization** is the process of converting high-precision numerical representations into lower-precision formats while attempting to preserve model accuracy.

The main objectives of quantization are:
- **Reducing memory bandwidth usage**
- **Increasing inference throughput**
- **Lowering latency**
- **Improving hardware utilization**
- **Reducing power consumption**

TensorRT applies quantization during **engine build time**, generating hardware-optimized inference engines tailored to the target GPU architecture.

---

## Precision Formats Explained

### **FP32 (Full Precision)**

**FP32** is the standard precision used during training. It provides:
- High numerical accuracy
- Strong stability during backpropagation

However, during inference it suffers from:
- High memory usage
- Lower computational efficiency
- Slower inference speed

FP32 is typically reserved for **training** or **accuracy-critical inference**.

---

### **FP16 (Half Precision)**

**FP16** reduces the precision of floating-point values to 16 bits. On modern NVIDIA GPUs, FP16 operations are accelerated using **Tensor Cores**.

Key properties:
- **Significant speedup** compared to FP32
- **Minimal or no accuracy loss** for most models
- Reduced memory footprint

FP16 is often the **default choice for production inference**.

---

### **INT8 (Integer Quantization)**

**INT8 quantization** converts weights and activations to 8-bit integers. This approach requires additional steps to preserve numerical accuracy.

Characteristics:
- **Lowest memory usage**
- **Highest throughput**
- **Lowest inference latency**

Trade-off:
- Potential **accuracy degradation** if not carefully calibrated

INT8 is ideal for **high-throughput, latency-sensitive applications**.

---

## ONNX in the Deployment Pipeline

![ONNX](/assets/img/posts/ONNX.jpg)
**ONNX (Open Neural Network Exchange)** is a **framework-agnostic intermediate representation** for deep learning models. It defines a standardized **computational graph**, allowing models to be exported from training frameworks and consumed by inference runtimes.

ONNX enables:
- **Model portability**
- **Cross-framework compatibility**
- **Runtime-level optimization**

**TensorRT natively consumes ONNX models**, making it the preferred format for optimized GPU inference.

---

## Quantization Strategies

There are two primary strategies for applying INT8 quantization:

---

### **Post-Training Quantization (PTQ)**

**Post-Training Quantization (PTQ)** applies quantization **after model training**, without modifying weights.

Technical process:
- Run a **calibration dataset** through the model
- Collect **activation statistics**
- Compute **scaling factors**
- Map floating-point tensors to **INT8 ranges**

## **Why 8-bit post-training quantization**:
The 8-bit quantization is just one of the available compression methods but one often selected for:
- significant performance results.
- little impact on accuracy.
- ease of use.
- wide hardware compatibility.


Advantages:
- No retraining required
- Fast deployment
- Low engineering complexity

Limitations:
- Accuracy depends heavily on **calibration data**
- Less effective for **deep or sensitive models**

PTQ is suitable when **training data is unavailable** or **rapid deployment is required**.

![ONNX](/assets/img/posts/quantization_picture.svg)

---

### **Quantization-Aware Training (QAT)**

**Quantization-Aware Training (QAT)** integrates quantization effects directly into the training process.

Technical process:
- Insert **fake quantization operators**
- Simulate **INT8 arithmetic** during forward passes
- Adjust weights via **backpropagation**

Advantages:
- **Higher INT8 accuracy**
- Reduced quantization error
- More stable activation distributions

Limitations:
- Requires **retraining**
- Increased training complexity
- Higher computational cost

QAT is preferred for **accuracy-critical production systems**.

---

## TensorRT and Quantization Workflows

TensorRT supports both PTQ and QAT through ONNX-based workflows.

### **PTQ Workflow**
1. Train model in **FP32**
2. Export model to **ONNX**
3. Apply **INT8 calibration** in TensorRT
4. Deploy optimized **INT8 inference engine**

### **QAT Workflow**
1. Train model using **quantization-aware training**
2. Export quantized model to **ONNX**
3. TensorRT preserves **quantization scales**
4. Deploy **high-accuracy INT8 engine**

---

## Performance Metrics

Inference performance is evaluated using two primary metrics:

- **Latency (ms):**  
  Time required to process a single input sample.

- **Throughput (FPS):**  
  Number of inferences processed per second.

Typical trends observed in benchmarks:
- **FP32** → Highest latency, lowest throughput
- **FP16** → Balanced performance and accuracy
- **INT8** → Lowest latency, highest throughput

---

## Key Takeaways

- **Quantization** is essential for efficient inference deployment
- **ONNX** enables seamless integration with TensorRT
- **FP16** provides an excellent speed–accuracy trade-off
- **INT8** delivers maximum performance with proper calibration
- **QAT** achieves superior accuracy compared to PTQ

---

## Conclusion

Quantization, when combined with **ONNX** and **TensorRT**, enables deep learning models to achieve **production-level performance** without sacrificing reliability. Understanding the trade-offs between **FP32**, **FP16**, and **INT8**, as well as choosing the appropriate quantization strategy, is a critical skill for modern AI engineers.
