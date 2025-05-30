# CGR

## 🎨 CGR: Content-Style Guided Representation & Transfer

CCSR (Conditional Content-Style Retrieval) is a multimodal retrieval framework that supports joint retrieval of **content and style** through a combination of text and image inputs. The project covers a complete pipeline including style encoder training, feature extraction, retrieval evaluation, and style transfer.

## **Environment Setup**

```bash
conda create -n ccsr python=3.8
conda activate ccsr
pip install -r requirements.txt
```

## 🚀 **Usage Workflow**

#### **Train Style Encoder**

```bash
python CGR/CSD/main_sim.py
```

Supports multi-GPU distributed training and various encoder backbones such as DINO, CLIP, MoCo, SSCD, and CSD.

#### **Extract Image Features**

```bash
python CGR/Classifier/Dataag_Getemb.py
```

Used for extracting embeddings for classification/retrieval tasks.

#### **Train and Evaluate Classifier**

```bash
python CGR/Classifier/Classificition_Test.py
```

#### **Build Feature Databases**

```bash
# Build content feature database
python CGR/Pipeline/buildcontentemb_database.py

# Build style feature database
python CGR/Pipeline/buildstyleemb_database.py
```

#### **Style Transfer Model**

```bash
cd CGR/FreeStyle/
python run_freestyle.py --content path/to/content.jpg --style path/to/style.jpg
```

## 🗂️ **StyleCoco Dataset**

We propose a new dataset called **StyleCoco**, which supports the evaluation of content-style joint retrieval. Its key features include:

- 80 content categories (from COCO)
- 27 style categories (from WikiArt)
- A total of 21,600 images with explicit annotations

Data construction: Style transfer methods (e.g., InsT) are used to fuse content and style images into labeled samples.

## 🌟 **Highlights**

- First to apply **contrastive learning** to conditional retrieval tasks
- Proposes a novel **Mixture-of-Experts (MoE)** system for expert collaboration
- Introduces **Prompt Learning** with a gating mechanism for adaptive control
- Outperforms previous methods across all three retrieval settings: content, style, and joint retrieval
