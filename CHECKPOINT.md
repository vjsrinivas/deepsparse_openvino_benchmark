# Models

* MobileViT:
    * MobileViT XXS
    * MobileViT XS
* EfficientViT:
    * EfficientViT_b0
    * EfficientViT_b2
* ConvNeXt:
    *
    * ConvNeXt-Tiny

# Model Properties Needed

Apply the following properties for each model:
* Sparsity
* Quantization to INT8
* Classification evaluation for sparsity + INT8

# CPU Machines (AWS):

## Amazon EC2 M5 Instances:
* M5 and M5d instances have Intel Xeon CPUs Platinum 8000 series (upto 3.1 GHz)
    * These CPUs have AVX-512 and VNNI extensions for ML acceleration 

### Instances:
* m5.xlarge
* m5.4xlarge
* m5d.large
* m5d.4xlarge

## Amazon EC2 C4 Instances:
* C4 instances have Intel Xeon processors (Haswell generation) (upto 2.9 GHz)

### Instances:
* c4.large
* c4.xlarge

---------------
* m5.xlarge [x]
    * CPU: Intel Xeon Platinum 8175 (3.1 GHz)
* m5n.xlarge [x]
    * CPU: 	Intel Xeon Platinum 8259 (Cascade Lake) (3.1 GHz)
* m6a.xlarge [x]
    * CPU: 	AMD EPYC 7R13 Processor (3.6 GHz)
* m7i.xlarge [x]
    * CPU: Intel Xeon Scalable (Sapphire Rapids) (3.2 GHz)
* c4.xlarge [x]
    * CPU: Intel Xeon E5-2666 v3 (Haswell) (2.9 GHz)
* t3.xlarge [x]
    * CPU: Intel Skylake E5 2686 v5 (3.1 GHz)
---------------

* inf1.xlarge [o] [Architecture geared towards Neuron SDK]
    * CPU: ntel Xeon Platinum 8275CL (Cascade Lake) (N/A GHz)


# Results

| Model Type | Pruned (Y/N) | Quantized (Y/N) |
| ---------- | ------------ | --------------- |
| efficientnet-b2 | Yes (51% Sparsity) | No (FLOAT32) |
| mobilenet_v1 | Yes (N/A Sparsity) | Yes (INT8) |
| resnet50 | Yes (N/A Sparsity) | Yes (INT8) |


# Sparsity Sensitivity:

**EfficientNet B2**
classifier -> 90
blocks.6.1.conv_pw -> 95
blocks.6.1.conv_pwl -> 95
conv_head -> 70
blocks.6.0.conv_pwl -> 50
blocks.5.1.conv_pw -> 95
blocks.5.1.conv_pwl -> 95
blocks.5.2.conv_pw -> 95
blocks.5.2.conv_pwl -> 95
blocks.5.3.conv_pw -> 95

**Acc:** 0.7444
**Sparsity:** 0.4839