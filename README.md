# 👗 Fashion Image Search with Picture Descriptions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-green.svg)](https://github.com/mlfoundations/open_clip)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning-based fashion image retrieval system that combines **visual similarity** with **natural language modifications**. Upload a reference image and describe the changes you want (e.g., *"make it sleeveless and blue"*) to find matching fashion items.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

This project implements a **composed image retrieval** system for fashion items. Unlike traditional image search that finds visually similar items, this system allows users to:

1. **Upload a reference image** (e.g., a red dress)
2. **Describe desired modifications** in natural language (e.g., *"is shorter and sleeveless"*)
3. **Retrieve images** that match both the visual features and the text description

The system leverages **OpenCLIP (ViT-B-32)** fine-tuned on the **Fashion-IQ dataset** to understand the relationship between fashion images and modification descriptions.

---

## ✨ Features

- **🖼️ + 📝 Combined Search**: Search using both image and text queries simultaneously
- **📝 Text-Only Search**: Find items using natural language descriptions alone
- **🖼️ Image-Only Search**: Traditional visual similarity search
- **⚖️ Adjustable Text Influence**: Control the balance between image and text features
- **🎨 Interactive Web Interface**: User-friendly Streamlit application
- **🔥 GPU Accelerated**: CUDA support for fast inference
- **📊 Similarity Scoring**: View confidence scores for each result

---

## 🏗️ Architecture

### Model Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    OpenCLIP ViT-B-32                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐         ┌─────────────┐                  │
│   │   Visual    │         │    Text     │                  │
│   │   Encoder   │         │   Encoder   │                  │
│   │  (ViT-B-32) │         │(Transformer)│                  │
│   └──────┬──────┘         └──────┬──────┘                  │
│          │                       │                          │
│          ▼                       ▼                          │
│   ┌─────────────┐         ┌─────────────┐                  │
│   │   Image     │         │    Text     │                  │
│   │ Embedding   │         │ Embedding   │                  │
│   │   (512-D)   │         │   (512-D)   │                  │
│   └──────┬──────┘         └──────┬──────┘                  │
│          │                       │                          │
│          └───────────┬───────────┘                          │
│                      ▼                                      │
│              ┌─────────────┐                                │
│              │  Combined   │                                │
│              │  Embedding  │                                │
│              │ (Additive)  │                                │
│              └──────┬──────┘                                │
│                     ▼                                       │
│              ┌─────────────┐                                │
│              │  Cosine     │                                │
│              │ Similarity  │                                │
│              └─────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### Training Approach

The model is trained using an **improved contrastive loss** that considers:

1. **Standard CLIP Loss**: Image-text alignment
2. **Composed Retrieval Loss**: Reference image + text modification → Target image
```python
# Combined query (reference image + text) should match target
combined_embedding = image_embedding + text_embedding
loss = CrossEntropyLoss(combined_query @ target_features.T, labels)
```

---

## 📊 Dataset

### Fashion-IQ Dataset

The project uses the [Fashion-IQ dataset](https://github.com/XiaoxiaoGuo/fashion-iq), which contains:

| Category | Train | Validation | Test |
|----------|-------|------------|------|
| Dress    | ~8,000 | ~2,000 | ~2,000 |
| Shirt    | ~8,000 | ~2,000 | ~2,000 |
| Top/Tee  | ~8,000 | ~2,000 | ~2,000 |

**Dataset Structure:**
- **Reference Image**: The starting point (candidate)
- **Target Image**: The desired result
- **Captions**: Natural language descriptions of differences (2 per pair)

**Example Entry:**
```json
{
  "target": "B008BHCT58",
  "candidate": "B003FGW7MK",
  "captions": [
    "is solid black with no sleeves",
    "is black with straps"
  ]
}
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/fashion-image-search.git
cd fashion-image-search
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision transformers matplotlib
pip install open_clip_torch
pip install streamlit pyngrok
pip install pillow tqdm pandas
```

### Step 4: Download Dataset
```bash
# Clone Fashion-IQ dataset
git clone https://github.com/XiaoxiaoGuo/fashion-iq.git
git clone https://github.com/hongwang600/fashion-iq-metadata.git
```

---

## 💻 Usage

### Running the Jupyter Notebook

1. Open `Fashion_Image_Search_final_code.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook will:
   - Download and preprocess the dataset
   - Train the model
   - Generate embeddings
   - Launch the Streamlit app

### Running the Streamlit App
```bash
streamlit run app.py
```

Or with ngrok for public access (in Colab):
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8501)
print(f"Access app at: {public_url}")
```

### Python API Usage
```python
import torch
import open_clip
from PIL import Image

# Load model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
state_dict = torch.load("path/to/openclip_model.pt")
model.load_state_dict(state_dict)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Encode image
image = preprocess(Image.open("reference.jpg")).unsqueeze(0)
image_features = model.encode_image(image)

# Encode text modification
text = tokenizer(["is shorter and sleeveless"])
text_features = model.encode_text(text)

# Combine for composed retrieval
combined = image_features + text_features
combined = combined / combined.norm(dim=-1, keepdim=True)
```

---

## 🏋️ Model Training

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | OpenCLIP ViT-B-32 |
| Batch Size | 8 |
| Epochs | 50 |
| Learning Rate | 5e-6 |
| Warmup Epochs | 5 |
| Weight Decay | 0.1 |
| Temperature | 0.07 |
| Optimizer | AdamW |

### Training Features

- **Data Augmentation**: Random horizontal flip, color jitter, affine transforms
- **Layer-wise Learning Rates**: Lower LR for pretrained layers
- **Cosine Annealing**: Learning rate schedule with warmup
- **Gradient Clipping**: Max norm = 1.0
- **Early Stopping**: Loss threshold at 0.1

---

## 📈 Results

### Training Performance

| Epoch | Loss | Status |
|-------|------|--------|
| 1 | 3.749 | Starting |
| 10 | 2.675 | Improving |
| 25 | 1.176 | Good |
| 40 | 0.770 | Excellent |
| 50 | 0.692 | **Best** |

### Retrieval Performance
```
Reference image: B0089I9NFS.jpg
Text query: 'is shorter and sleeveless'

Top 5 Results (Image + Text Combined):
1. ID: B0089I9NFS | Similarity: 0.7611 ✓ Good
2. ID: B004BX98O6 | Similarity: 0.5914 ○ OK
3. ID: B00385WM5A | Similarity: 0.5842 ○ OK
4. ID: B008I2VXU8 | Similarity: 0.5228 ○ OK
5. ID: B0075NK230 | Similarity: 0.5137 ○ OK
```

---

## 📁 Project Structure
```
fashion-image-search/
│
├── Fashion_Image_Search_final_code.ipynb  # Main notebook
├── app.py                                  # Streamlit application
├── requirements.txt                        # Dependencies
├── README.md                               # This file
│
├── fashion-iq/                             # Dataset directory
│   ├── captions/
│   │   ├── cap.dress.train.json
│   │   ├── cap.train_split.json
│   │   └── cap.test_split.json
│   ├── image_splits/
│   └── annotations/
│
├── fashion-iq-metadata/
│   └── image_url/
│       ├── asin2url.dress.txt
│       ├── downloaded_images/
│       └── downloaded_images_short/
│
└── OUTPUT/
    ├── openclip_model.pt                   # Trained model weights
    ├── image_embeddings.pkl                # Pre-computed embeddings
    ├── image_metadata.json                 # Image metadata
    └── training_history.json               # Training logs
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework |
| **OpenCLIP** | CLIP model implementation |
| **Transformers** | Tokenization |
| **Streamlit** | Web interface |
| **PIL/Pillow** | Image processing |
| **NumPy** | Numerical operations |
| **Pandas** | Data manipulation |
| **tqdm** | Progress bars |
| **ngrok** | Public URL tunneling |

---

## 🔮 Future Improvements

- [ ] Train on full Fashion-IQ dataset with all categories (dress, shirt, toptee)
- [ ] Experiment with larger models (ViT-L-14, ViT-H-14)
- [ ] Implement attention-based fusion instead of additive composition
- [ ] Add hard negative mining for improved training
- [ ] Deploy as a production-ready Docker container
- [ ] Build mobile application

---

## 👤 Author

**Srivalli Vangaveti**

---

## 🙏 Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP) - Original CLIP implementation
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open-source CLIP reproduction
- [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq) - Dataset and benchmark
- [Streamlit](https://streamlit.io/) - Web application framework

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

</div>
