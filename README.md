# codsoft-task3
# Image Captioning AI

A deep learning application that generates natural language captions for images using the Salesforce BLIP model. Built as Task 3 of the CodSoft AI Internship.

---

## Overview

This project combines Computer Vision and Natural Language Processing to automatically describe the contents of any image. The user uploads a photo through a web interface, and the model generates one or more captions describing what it sees — along with a confidence score for each.

---

## Features

- Generate 1 to 3 caption variations for any image
- Four caption styles: Default, Detailed, Short, and Story-like
- Adjustable caption length from 20 to 100 tokens
- Confidence score displayed alongside each caption
- Download generated captions as a .txt file
- Public shareable link via Gradio Live

---

## How It Works

The app uses Salesforce BLIP (Bootstrapping Language-Image Pre-training), a vision-language model pre-trained on large image-text datasets.

1. The uploaded image is preprocessed using the BLIP processor
2. A Vision Transformer (ViT) encoder extracts visual features from the image
3. A BERT-based text decoder generates captions from those features, one token at a time
4. Beam search with 5 beams finds the most probable caption sequences
5. Confidence scores are derived from the beam scores and normalized to a 60–99% range for readability

---

## Project Structure

```
image-captioning-ai/
│
├── app.py              # Main application — model loading, caption logic, and UI
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes cache, venv, and Gradio temp files
└── README.md           # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-captioning-ai.git
cd image-captioning-ai

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

The app opens at `http://127.0.0.1:7860`. A public Gradio Live link is also printed in the terminal for sharing.

> Note: The first run downloads the BLIP model weights (~900MB). This is a one-time download — subsequent runs load from local cache and start instantly.

---

## Dependencies

```
transformers
torch
torchvision
pillow
gradio
```

---

## Model Details

| Property | Value |
|----------|-------|
| Model | Salesforce/blip-image-captioning-base |
| Architecture | ViT encoder + BERT-based decoder |
| Task | Image-to-text generation |
| Inference | Beam search, 5 beams |
| Source | HuggingFace Transformers |

---

## Caption Styles

| Style | Behaviour |
|-------|-----------|
| Default | Straightforward, natural description |
| Detailed | Longer output with more scene context |
| Short | Concise single-line caption |
| Story-like | Narrative and creative tone |

---

## Internship Context

| Detail | Info |
|--------|------|
| Internship | CodSoft AI Internship |
| Task | Task 3 — Image Captioning |
| Task 1 | Rule-Based Chatbot |
| Task 2 | Recommendation System |
| Duration | June 2026 |

---

## Author

Made by Baby Sree as a part of the AI Internship at CodSoft.

