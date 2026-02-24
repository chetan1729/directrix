# --- 1. HARDWARE LOCK ---
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import keras_hub
import pickle
from datetime import datetime

def log_event(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Demo: {message}")

PRESET_PATH = "/workspace/gemma_keras_preset"
CHECKPOINT_PATH = "checkpoints/ckpt_step_7000.pkl"

SCENARIOS = [
    "Give me career advice for a junior software engineer entering the field of LLM alignment.",
    "Explain quantum entanglement to a 10-year-old using a simple analogy.",
    "I am struggling to stay productive while working from home. What are three concrete steps I can take today?"
]

def run_demo():
    log_event("Starting Side-by-Side Demo...")
    
    # Load models
    log_event("Loading Base Model...")
    base_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
    
    log_event("Loading Tuned Model (Step 7000)...")
    tuned_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
    tuned_model.backbone.enable_lora(rank=8)
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    for v, val in zip(tuned_model.trainable_variables, ckpt['p_vars']):
        v.assign(val)

    print("\n" + "!"*100)
    print(f"{'BASE MODEL':<48} | {'GEMMA-2B-DIRECTRIX (STEP 7000)':<48}")
    print("!"*100)

    for q in SCENARIOS:
        print(f"\nPROMPT: {q}")
        print("-" * 100)
        
        base_out = base_model.generate(q, max_length=200).replace("\n", " ")
        tuned_out = tuned_model.generate(q, max_length=200).replace("\n", " ")
        
        # Simple wrap for side-by-side
        for i in range(0, 400, 45):
            b_chunk = base_out[i:i+45].ljust(45)
            t_chunk = tuned_out[i:i+45].ljust(45)
            if b_chunk.strip() or t_chunk.strip():
                print(f"{b_chunk} | {t_chunk}")
        print("-" * 100)

    log_event("Demo Complete.")

if __name__ == "__main__":
    run_demo()
