# AI Navigation Assistive System for Visually Impaired Individuals

A real-time computer vision-based navigation system designed to assist visually impaired individuals in urban environments by detecting walkable areas and obstacles.

## Features

- **Semantic Segmentation:** Fine-tuned `SegFormer-B2` model trained on the SideGuide dataset for high-accuracy path and obstacle identification.
- **Real-Time Processing:** Optimized image processing pipeline built with `Cython`, achieving **30 FPS** on GPU.
- **Audio Feedback:** Real-time generation of spatial audio instructions to describe the user's surroundings.

## Tech Stack

- **Languages:** Python, Cython
- **Deep Learning Framework:** PyTorch, Hugging Face Transformers
- **Computer Vision:** OpenCV, SegFormer

## Project Architecture

1. **Input:** Live video feed from wearable or mobile camera.
2. **Segmentation:** `SegFormer-B2` segments navigation zones and potential obstacles.
3. **Pipeline Optimization:** Cython-accelerated frame processing for low-latency performance.
4. **Output:** Real-time audio cues directing the user safely.

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
