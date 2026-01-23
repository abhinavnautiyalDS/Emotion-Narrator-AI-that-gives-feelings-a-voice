# 🧠 Emotion Narrator  
**A Multimodal AI System that Sees, Understands, and Speaks Human Emotion**
<img width="1145" height="687" alt="emotion" src="https://github.com/user-attachments/assets/6c4ca626-22ed-448a-85b5-20b79b4adeaf" />



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

## Deployment


- Fine-tuned LLM hosted on Hugging Face Hub

- Streamlit used for real-time inference

- Modular architecture enables easy upgrades




https://github.com/user-attachments/assets/2a006b2b-6581-4fd3-b407-93fe3145e610




## Key Learnings

- Data quality matters more than model complexity

- Transfer learning is essential for small datasets

- Emotion detection ≠ emotion understanding

- Multimodal systems outperform single-model approaches

- Responsible AI requires uncertainty handling

## Future Improvements

- Multilingual narration

- Emotion intensity scaling

- Advanced neural TTS

- Video-based emotion analysis

- Temporal emotion tracking

## 📖 Detailed Explanation (Medium Blog)

For a complete in-depth explanation of the project — including:

- CNN design and EfficientNet architecture
- Dataset challenges and how they were solved
- LLM fine-tuning strategy
- Prompt engineering and narration logic
- Multimodal pipeline design
- Deployment approach

👉 Read the full technical walkthrough on Medium:  
**Emotion Narrator — Building a Multimodal AI System that Sees, Understands, and Speaks Human Emotion**

🔗 *Medium link:*  
https://medium.com/@abhinavnautiyal96/emotion-narrator-building-a-multimodal-ai-system-that-sees-understands-and-speaks-human-emotion-9914b410da35?postPublishedType=initial

