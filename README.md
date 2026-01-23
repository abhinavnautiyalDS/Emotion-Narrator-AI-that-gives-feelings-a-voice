# 🧠 Emotion Narrator  
**A Multimodal AI System that Sees, Understands, and Speaks Human Emotion**

Emotion Narrator is a multimodal artificial intelligence system designed to go beyond traditional facial emotion recognition.  
Instead of stopping at emotion labels, the system interprets emotional signals and expresses them through natural language narration and voice output.

---

## 🚀 Project Overview

Most facial emotion recognition systems return outputs such as *happy*, *sad*, or *angry*.  
While technically correct, these predictions lack emotional meaning for real users.

Emotion Narrator introduces an interpretation layer that converts emotional signals into:

- empathetic narration
- contextual understanding
- voice-based emotional expression

---

## 🔗 System Pipeline

<img width="844" height="243" alt="image" src="https://github.com/user-attachments/assets/86c7bbfb-d548-42c5-b777-3b52930ecf94" />


---

## 🧩 Core Components

### 1️⃣ CNN — Emotion Extraction (Vision Layer)

- Implemented using **transfer learning**
- Backbone: **EfficientNet (pretrained on ImageNet)**
- Dataset used: **TED-EEF (TDEF) facial emotion dataset**
- Manual dataset structuring into train/test folders
- Pretrained layers frozen to preserve visual features
- Custom classification head trained for emotion prediction

**Output:**
- emotion probabilities
- dominant emotion
- confidence score

The CNN focuses only on perception, not interpretation.

---

### 2️⃣ LLM — Emotional Interpretation (Reasoning Layer)

- Base model: **DistilGPT-2**
- Fine-tuned using **Supervised Fine-Tuning (SFT)**
- Input includes:
  - detected emotion
  - confidence score
  - time of day
  - profession
  - environment
- Output: empathetic narration (4–6 sentences)

**Key design decisions:**
- prompt tokens masked during loss calculation
- narration-only learning
- confidence-aware narration to reduce hallucination
- consistent narration tone

The LLM converts structured emotional signals into meaningful human language.

---

### 3️⃣ TTS — Voice Expression (Expression Layer)

- Converts narration text into speech
- Enhances emotional realism
- Completes the emotional interaction loop

---

## 🧠 Design Philosophy

Emotion Narrator follows a human-inspired cognitive structure:

| Human Cognition | System Module |
|----------------|----------------|
| Eyes           | CNN |
| Brain          | LLM |
| Voice          | TTS |

Each module has a single responsibility, making the system modular and scalable.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Streamlit
- OpenCV
- NumPy
- gTTS
- Hugging Face Hub

---

## 📁 Project Structure

Emotion-Narrator/
│
├── README.md
│
├── app/
│   ├── streamlit_app.py          # Main application pipeline
│   ├── utils/
│   │   ├── cnn_utils.py          # CNN preprocessing & prediction
│   │   ├── llm_utils.py          # Prompt building & generation
│   │   ├── tts_utils.py          # Text-to-speech logic
│   │   └── image_utils.py        # Face detection & preprocessing
│
├── models/
│   ├── cnn/
│   │   └── emotional_narrator_CNN.keras
│   │
│   ├── llm/
│   │   └── distilgpt2_finetuned/
│   │
│   └── tts/
│       └── (optional config)
│
├── training/
│   ├── cnn_training.ipynb        # CNN training notebook
│   ├── llm_finetuning.ipynb      # LLM fine-tuning notebook
│   └── train_cpu_sft_masked.py   # SFT training script
│
├── data/
│   ├── tdef_dataset/
│   │   ├── train/
│   │   │   ├── angry/
│   │   │   ├── happy/
│   │   │   ├── sad/
│   │   │   └── ...
│   │   └── test/
│   │       ├── angry/
│   │       ├── happy/
│   │       ├── sad/
│   │       └── ...
│   │
│   └── llm_training_data/
│       └── emotion_narration.jsonl
│
├── assets/
│   ├── pipeline_diagram.png
│   ├── app_screenshot.png
│   └── dataset_samples.png
│
├── requirements.txt
│
└── .env



