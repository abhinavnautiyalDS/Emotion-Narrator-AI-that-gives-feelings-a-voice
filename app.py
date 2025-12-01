# loads .env into os.environ
import os
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = 'abhinav965108/distilgpt2_model'
HF_SUBFOLDER = 'cpu-sft-distil/cpu-sft-distil'
KERAS_MODEL_PATH = "emotional_narrator_CNN_part1.keras"  # Yeh file GitHub mein honi chahiye
import streamlit as st
from PIL import Image
import base64
import json
import time
from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import PIL.ImageStat as Stat
import torch
from huggingface_hub import login as hf_login
import cv2

# Set page configuration to ensure wide mode and no default padding
st.set_page_config(
    page_title="Emotion Narrator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to achieve the Cyberpunk Look
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

    /* Global Reset & Background */
    .stApp {
        background-color: #061028;
        background-image: 
            linear-gradient(rgba(6, 16, 40, 0.95), rgba(6, 16, 40, 0.95)),
            linear-gradient(0deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        color: #d1f7ff;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 90rem;
    }

    /* Headers */
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
        margin-bottom: 3rem;
    }

    h3 {
        font-family: 'Orbitron', sans-serif;
        color: rgba(0, 255, 255, 0.9);
        margin-bottom: 1rem;
    }

    /* Image Container Frame */
    .image-frame {
        position: relative;
        padding: 4px;
        border: 1px solid rgba(0, 255, 255, 0.3);
        background: rgba(6, 16, 40, 0.5);
    }
    
    .image-frame::before, .image-frame::after {
        content: '';
        position: absolute;
        width: 20px;
        height: 20px;
        border: 2px solid #00ffff;
        transition: all 0.3s;
    }
    
    .image-frame::before { top: -2px; left: -2px; border-right: 0; border-bottom: 0; }
    .image-frame::after { bottom: -2px; right: -2px; border-left: 0; border-top: 0; }

    /* Stats Panel Styling */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 12px;
        margin-bottom: 8px;
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.2);
        align-items: center;
    }

    .stat-label {
        color: #88ccff;
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    .stat-value {
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stat-highlight {
        color: #00ffff;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    }

    /* File Uploader Styling */
    .stFileUploader > div > small {
        display: none;
    }
    
    .stFileUploader button {
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
        background: rgba(0, 255, 255, 0.05) !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Waveform Animation Placeholder styling */
    .waveform-container {
        margin-top: 3rem;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 50px;
        background: rgba(0, 255, 255, 0.02);
        position: relative;
    }
    
    .bar {
        width: 6px;
        margin: 0 3px;
        background: linear-gradient(to top, rgba(0,255,255,0.2), #00ffff);
        border-radius: 3px;
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    @keyframes pulse {
        0% { height: 20%; opacity: 0.5; }
        50% { height: 80%; opacity: 1; box-shadow: 0 0 10px #00ffff; }
        100% { height: 20%; opacity: 0.5; }
    }

    /* Button Styling */
    .stButton button {
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
        background: rgba(0, 255, 255, 0.05) !important;
        font-family: 'Orbitron', sans-serif !important;
        width: 100%;
        padding: 10px;
        margin-top: 20px;
    }

    /* Input Field Styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: rgba(0, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #88ccff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

</style>
""", unsafe_allow_html=True)

# =====================================================================
# BACKEND FUNCTIONS
# =====================================================================

OUT_DIR = Path('emotion_narrator_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FER_CLASSES = ['angry','disgust','fear','happy','neutral','sad','surprise']
CUSTOM_PREPROCESS = None

# Initialize models
@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    models = {}
    
    # Load HuggingFace model
    try:
        if HF_SUBFOLDER:
            tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, subfolder=HF_SUBFOLDER, use_fast=True)
            hf_model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID, subfolder=HF_SUBFOLDER)
        else:
            tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, use_fast=True)
            hf_model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            hf_model.resize_token_embeddings(len(tokenizer))

        hf_model.config.pad_token_id = tokenizer.pad_token_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model.to(device)
        models['tokenizer'] = tokenizer
        models['hf_model'] = hf_model
        ##st.success(f"HF model loaded on: {device}")
    except Exception as e:
        st.error(f"HF model load error: {e}")
        return None

    # Load Keras CNN model - YEH IMPORTANT CHANGE HAI
    try:
        if os.path.exists(KERAS_MODEL_PATH):
            cnn_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
            models['cnn_model'] = cnn_model
            ##st.success("CNN model loaded successfully")
        else:
            # Agar file nahi mili toh simple model bana dete hain
            st.warning(f"Keras model not found at: {KERAS_MODEL_PATH}")
            st.warning("Using fallback emotion detection")
            models['cnn_model'] = "fallback"  # String mark kar dete hain
    except Exception as e:
        st.error(f"CNN model load error: {e}")
        models['cnn_model'] = "fallback"  # String mark kar dete hain

    return models

def parse_time_input(value):
    if value is None:
        raise ValueError("No time provided")
    raw = str(value).strip().lower()
    if re.fullmatch(r"\d{1,2}$", raw):
        h = int(raw)
        if 0 <= h <= 23:
            hour = h
            minute = 0
            ampm = 'am' if hour < 12 else 'pm'
        else:
            hour = 12 if h == 12 else h % 12
            ampm = 'pm' if h == 12 else 'am'
            minute = 0
        return hour, minute, ampm, hour_to_period(hour)
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", raw)
    if m:
        h = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        suffix = m.group(3)
        if suffix == 'am':
            hour = 0 if h == 12 else h
            ampm = 'am'
        elif suffix == 'pm':
            hour = 12 if h == 12 else h + 12
            ampm = 'pm'
        else:
            hour = h
            ampm = 'am' if h < 12 else 'pm'
        return hour, minute, ampm, hour_to_period(hour)
    raise ValueError(f"Invalid time format: {value}")

def hour_to_period(h):
    h = h % 24
    if 5 <= h <= 11: return "morning"
    if 12 <= h <= 16: return "afternoon"
    if 17 <= h <= 20: return "evening"
    return "night"

def model_uses_efficientnet(model):
    if isinstance(model, str) and model == "fallback":
        return False
    try:
        ishape = getattr(model, 'input_shape', None)
        if ishape and len(ishape) == 4:
            _, h, w, c = ishape
            if int(h) == 260 and int(w) == 260:
                return True
    except Exception:
        pass
    for layer in (model.layers[:6] if hasattr(model, 'layers') else []):
        name = getattr(layer, 'name', "")
        if "efficientnet" in name.lower() or "efficient" in name.lower():
            return True
    return False

def preprocess_for_model(pil_img, model=None):
    if CUSTOM_PREPROCESS is not None:
        return CUSTOM_PREPROCESS(pil_img)
    target_h, target_w, channels = 224, 224, 3
    try:
        if model is not None and model != "fallback" and hasattr(model, 'input_shape') and model.input_shape:
            ishape = model.input_shape
            if len(ishape) == 4:
                target_h = int(ishape[1]) or target_h
                target_w = int(ishape[2]) or target_w
                channels = int(ishape[3]) if ishape[3] else channels
    except Exception:
        pass
    if channels == 1:
        img = pil_img.convert('L')
    else:
        img = pil_img.convert('RGB')
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img).astype('float32')
    try:
        if model is not None and model != "fallback" and model_uses_efficientnet(model):
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
            batch = np.expand_dims(arr, 0) if arr.ndim == 3 else arr
            return eff_pre(batch)
    except Exception as e:
        print("EfficientNet preprocess failed, fallback to /255:", e)
    if arr.ndim == 3:
        arr = arr / 255.0
        arr = np.expand_dims(arr, 0)
    elif arr.ndim == 2:
        arr = np.expand_dims(arr, (0, -1))
        arr = arr / 255.0
    return arr

def detect_face_pil(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces)==0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces[0]
    face = bgr[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

def get_cnn_probs(model, img):
    # YEH FIX HAI - agar model "fallback" hai toh random probabilities return karo
    if isinstance(model, str) and model == "fallback":
        # Simple fallback based on image properties
        img_array = np.array(img.convert('L'))
        brightness = np.mean(img_array)
        
        # Brightness ke basis pe emotion predict karo
        if brightness > 200:
            # Very bright - happy/surprise
            probs = np.array([0.1, 0.0, 0.0, 0.6, 0.1, 0.1, 0.1])
        elif brightness > 150:
            # Bright - happy/neutral
            probs = np.array([0.05, 0.0, 0.0, 0.4, 0.4, 0.05, 0.1])
        elif brightness > 100:
            # Medium - neutral
            probs = np.array([0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.1])
        elif brightness > 50:
            # Dark - sad/angry
            probs = np.array([0.3, 0.1, 0.1, 0.05, 0.1, 0.3, 0.05])
        else:
            # Very dark - sad/fear
            probs = np.array([0.2, 0.1, 0.3, 0.0, 0.05, 0.3, 0.05])
            
        return probs / probs.sum()
    
    # Agar actual model hai toh uska predict use karo
    x = preprocess_for_model(img, model)
    logits = model.predict(x)[0]
    return tf.nn.softmax(logits).numpy()

def fuse_predictions(original, face, mirror, weight_face=0.9):
    tta = (face + mirror) / 2.0
    fused = weight_face * tta + (1.0 - weight_face) * original
    fused = fused / fused.sum()
    idx = int(np.argmax(fused))
    return FER_CLASSES[idx], float(fused[idx]), fused, {
        "original": original.tolist(),
        "face": face.tolist(),
        "face_mirror": mirror.tolist(),
        "face_tta_avg": tta.tolist(),
        "final": fused.tolist()
    }

def predict_emotion_fused(pil_img, model):
    orig = get_cnn_probs(model, pil_img)
    face = detect_face_pil(pil_img)
    if face is None:
        idx = int(np.argmax(orig))
        return FER_CLASSES[idx], float(orig[idx]), orig, {"original": orig.tolist(), "final": orig.tolist()}
    face_probs = get_cnn_probs(model, face)
    mirror = face.transpose(Image.FLIP_LEFT_RIGHT)
    mirror_probs = get_cnn_probs(model, mirror)
    return fuse_predictions(orig, face_probs, mirror_probs)

def compute_facial_descriptors(pil_img: Image.Image):
    img = pil_img.convert('RGB')
    w,h = img.size
    stat = Stat.Stat(img)
    brightness = float(sum(stat.mean)/len(stat.mean))
    contrast = float(sum(stat.rms)/len(stat.rms))
    top = img.crop((0,0,w,int(h/3))).convert('L')
    mid = img.crop((0,int(h/3),w,int(2*h/3))).convert('L')
    bot = img.crop((0,int(2*h/3),w,h)).convert('L')
    top_mean = Stat.Stat(top).mean[0]
    mid_mean = Stat.Stat(mid).mean[0]
    bot_mean = Stat.Stat(bot).mean[0]
    eyebrow_tension = max(0.0, (mid_mean - top_mean) / 255.0)
    mouth_variance = Stat.Stat(bot).var[0]
    mouth_open = float(min(1.0, mouth_variance / (255.0*255.0) * 50.0))
    return {
        'brightness': brightness,
        'contrast': contrast,
        'top_mean': top_mean,
        'mid_mean': mid_mean,
        'bot_mean': bot_mean,
        'eyebrow_tension': eyebrow_tension,
        'mouth_open_proxy': mouth_open
    }

def build_prompt(time, age, profession, environment, emotion, confidence, extra_hint="none"):
    start_instr = ""
    try:
        if float(confidence) < 0.4:
            start_instr = "Start the narration with the single word 'Possibly'.\n"
    except Exception:
        start_instr = ""
    prompt = (
        f"Time: {time}\n"
        f"Age: {age}\n"
        f"Profession: {profession}\n"
        f"Environment: {environment}\n"
        f"Dominant emotion: {emotion}\n"
        f"Confidence: {confidence}\n"
        f"Extra hint: {extra_hint}\n\n"
        f"{start_instr}"
        "Write a 4-6 sentence empathetic narration in British English. Do NOT diagnose; if confidence < 0.4 start with 'Possibly'.\n"
    )
    return prompt

def generate_narration_from_fields(tokenizer, hf_model, time, age, profession, environment, emotion, confidence, max_new_tokens=150):
    prompt = build_prompt(time, age, profession, environment, emotion, confidence)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs["input_ids"].to(hf_model.device)
    attention_mask = inputs.get("attention_mask").to(hf_model.device) if "attention_mask" in inputs else None

    generated = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        num_return_sequences=1,
        use_cache=True,
    )

    gen_ids = generated[0]
    prompt_len = input_ids.shape[1]
    text_out = ""
    if gen_ids.shape[0] > prompt_len:
        new_ids = gen_ids[prompt_len:]
        text_out = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if not text_out:
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        try:
            decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if raw.startswith(decoded_prompt):
                text_out = raw[len(decoded_prompt):].strip()
            else:
                text_out = raw.strip()
        except Exception:
            text_out = raw.strip()
    return text_out

def generate_audio(narration_text):
    """Generate audio from narration text using gTTS"""
    try:
        audio_path = OUT_DIR / f"narr_{int(time.time())}.mp3"
        tts = gTTS(text=narration_text, lang='en')
        tts.save(str(audio_path))
        return audio_path
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None

# =====================================================================
# FRONTEND UI
# =====================================================================

# Title
st.markdown("<h1>EMOTION NARRATOR</h1>", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'narration_generated' not in st.session_state:
    st.session_state.narration_generated = False
if 'narration_text' not in st.session_state:
    st.session_state.narration_text = ""
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'emotion_result' not in st.session_state:
    st.session_state.emotion_result = None

# Load models
with st.spinner("Loading AI models..."):
    models = load_models()

if models:
    st.session_state.models_loaded = True
    if models['cnn_model'] == "fallback":
        st.warning("⚠️ CNN model not found. Using fallback emotion detection (less accurate).")
    else:
        st.success("All models loaded successfully!")

# Layout: Two main columns for content
col1, col2 = st.columns([1.2, 1])

with col1:
    # Image Upload Section
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width =True)
        st.session_state.uploaded_image = image
    else:
        # Placeholder if no image
        st.markdown("""
        <div style="height: 400px; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); color: #00ffff; font-family: 'Orbitron';">
            [ UPLOAD PHOTO TO BEGIN ]
        </div>
        """, unsafe_allow_html=True)
        st.session_state.uploaded_image = None
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Custom Upload Button Text
    st.markdown("""
    <div style="text-align: center; margin-top: 10px; color: #00ffff; font-family: 'Orbitron'; font-size: 0.8rem;">
        [ CLICK BROWSE FILES ABOVE ]
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Personal Data Section
    st.markdown("""
    <div style="border: 1px solid rgba(0,255,255,0.3); padding: 15px; background: rgba(0,255,255,0.05); margin-bottom: 20px; position: relative;">
        <div style="position: absolute; top: 0; right: 0; width: 30px; height: 30px; background: linear-gradient(225deg, rgba(0,255,255,0.2), transparent);"></div>
        <h3 style="margin: 0;">Personal Data</h3>
    </div>
    """, unsafe_allow_html=True)

    # Editable input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=25, key="age_input")
    profession = st.text_input("Profession", value="Software Engineer", key="profession_input")
    daytime = st.text_input("Daytime", value="Afternoon", key="time_input", 
                          help="Enter time (e.g., '2pm', '14:00', 'morning', 'afternoon')")

    # Generate Narration Button
    if st.button("Generate Narration", type="primary"):
        if not st.session_state.models_loaded:
            st.error("Models are still loading. Please wait...")
        elif st.session_state.uploaded_image is None:
            st.error("Please upload an image first.")
        else:
            with st.spinner("Analyzing emotion and generating narration..."):
                try:
                    # Emotion detection
                    emotion_label, confidence, fused_vector, fusion_debug = predict_emotion_fused(
                        st.session_state.uploaded_image, 
                        models['cnn_model']
                    )
                    
                    # Parse time input
                    try:
                        h, m, ampm, period = parse_time_input(daytime)
                        time_str = f"{h:02d}:{m:02d} {ampm.upper()}"
                    except:
                        time_str = daytime
                        period = daytime.lower()
                    
                    # Generate narration
                    narration = generate_narration_from_fields(
                        models['tokenizer'],
                        models['hf_model'],
                        time_str,
                        age,
                        profession,
                        period,
                        emotion_label,
                        confidence
                    )
                    
                    # Generate audio
                    audio_path = generate_audio(narration)
                    
                    # Update session state
                    st.session_state.narration_generated = True
                    st.session_state.narration_text = narration
                    st.session_state.audio_path = audio_path
                    st.session_state.emotion_result = {
                        'emotion': emotion_label,
                        'confidence': confidence,
                        'time': time_str,
                        'period': period
                    }
                    
                    st.success("Narration generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

    # Display emotion results if available
    if st.session_state.emotion_result:
        emotion_data = st.session_state.emotion_result
        data = [
            ("Age", str(age), False),
            ("Profession", profession, False),
            ("Daytime", emotion_data['period'].title(), False),
            ("Detected Emotion", f"{emotion_data['emotion'].title()} ({emotion_data['confidence']:.2f})", True)
        ]

        for label, value, highlight in data:
            highlight_class = "stat-highlight" if highlight else ""
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value {highlight_class}">{value}</span>
            </div>
            """, unsafe_allow_html=True)

# Narration Output Section
if st.session_state.narration_generated:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-family: 'Orbitron'; color: rgba(0,255,255,0.8); letter-spacing: 2px; margin-bottom: 10px;">
            AI NARRATION OUTPUT
        </div>
    """, unsafe_allow_html=True)
    
    # Display narration text
    st.markdown(f"""
    <div style="border: 1px solid rgba(0,255,255,0.3); padding: 20px; background: rgba(0,255,255,0.05); margin-bottom: 20px; color: #ffffff; font-family: 'Rajdhani'; line-height: 1.6;">
        {st.session_state.narration_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Play audio if available
    if st.session_state.audio_path and st.session_state.audio_path.exists():
        with open(st.session_state.audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3",autoplay=True)
    else:
        # Fallback waveform animation
        bars_html = "".join([f'<div class="bar" style="height: {30 + (i%5)*10}%; animation-delay: -{i*0.1}s"></div>' for i in range(40)])
        st.markdown(f"""
        <div class="waveform-container">
            {bars_html}
        </div>
        """, unsafe_allow_html=True)
else:
    # Default waveform placeholder when no narration
    bars_html = "".join([f'<div class="bar" style="height: {30 + (i%5)*10}%; animation-delay: -{i*0.1}s"></div>' for i in range(40)])
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-family: 'Orbitron'; color: rgba(0,255,255,0.8); letter-spacing: 2px; margin-bottom: 10px;">
            AI NARRATION OUTPUT
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="waveform-container">
        {bars_html}
    </div>
    """, unsafe_allow_html=True)

# Clear button to reset
if st.session_state.narration_generated:
    if st.button("Clear Results"):
        st.session_state.narration_generated = False
        st.session_state.narration_text = ""
        st.session_state.audio_path = None
        st.session_state.emotion_result = None
        st.rerun()
