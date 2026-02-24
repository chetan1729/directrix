import os
import jax
import datetime
from huggingface_hub import snapshot_download

# Log: Sprint setup started at {datetime.datetime.now()}
def log_event(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log: {message}")

# --- AUTH & DOWNLOAD ---
MY_TOKEN = "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" 
REPO_ID = "google/gemma-1.1-2b-it-keras" 
LOCAL_DIR = "/workspace/gemma_keras_preset"

if not os.path.exists(LOCAL_DIR):
    log_event(f"Downloading Keras-native preset to {LOCAL_DIR}...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        token=MY_TOKEN,
    )
    log_event("Local preset ready.")
else:
    log_event("Found existing preset, skipping download.")

# --- BACKEND CONFIG ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras
import keras_hub

# Hardware Handshake
devices = jax.devices()
log_event(f"Active Devices - {devices}")

# --- MODEL LOADING ---
PRESET_PATH = "/workspace/gemma_keras_preset"

try:
    log_event(f"Loading from: {PRESET_PATH}")
    # Loading into JAX memory
    model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
    ref_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
    
    ref_model.trainable = False
    log_event("SUCCESS: Models are live in VRAM on the 5090 cluster.")
    
    # Simple log putting: verification
    response = model.generate("What is your current alignment status?", max_length=32)
    print(f"\nModel Response: {response}\n")

except Exception as e:
    log_event(f"Load failed: {e}")
    if os.path.exists(PRESET_PATH):
        log_event(f"Contents of {PRESET_PATH}: {os.listdir(PRESET_PATH)}")