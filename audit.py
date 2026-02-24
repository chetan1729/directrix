# --- 1. HARDWARE LOCK ---
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import keras_hub
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def log_event(message, also_to_file=True):
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted_msg = f"[{timestamp}] Audit: {message}"
    print(formatted_msg)
    if also_to_file:
        with open("logs/audit.log", "a") as f:
            f.write(formatted_msg + "\n")

PRESET_BASE = "/workspace/gemma_keras_preset"
PRESET_MERGED = "/workspace/gemma_2b_directrix_final"

QUESTIONS = [
    "Python: Write a thread-safe Singleton implementation using a decorator or metaclass.",
    "Logic: If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Explain.",
    "Physics: Explain the formation of the depletion region in a P-N junction and its role in diode operation.",
    "Coding: Implement a binary search algorithm in JAX using jax.lax.while_loop.",
    "Reasoning: Brothers and sisters, I have none. But that man's father is my father's son. Who is it?"
]

def run_audit():
    log_event("--- STARTING FINAL PROJECT AUDIT ---")

    # 1. Audit Base Model
    log_event("Loading Base Model (Gemma-2B Raw)...")
    base_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_BASE)
    base_results = []
    for q in QUESTIONS:
        log_event(f"Querying Base: {q[:25]}...")
        prompt = f"\n\nHuman: {q}\n\nAssistant:"
        res = base_model.generate(prompt, max_length=256)
        base_results.append(res)
    
    del base_model
    jax.clear_caches()

    # 2. Audit Your Merged Model
    log_event("Loading Merged Model (Gemma-2B-Directrix)...")
    merged_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_MERGED)
    merged_results = []
    for q in QUESTIONS:
        log_event(f"Querying Merged: {q[:25]}...")
        prompt = f"\n\nHuman: {q}\n\nAssistant:"
        res = merged_model.generate(prompt, max_length=256)
        merged_results.append(res)

    # 3. Final Comparison Report & Logging
    log_event("Generating Comparison Report...")
    report_header = "\n" + "="*80 + "\nFINAL AUDIT REPORT: BASE vs. DIRECTRIX\n" + "="*80
    print(report_header)
    
    with open("logs/audit.log", "a") as f:
        f.write(report_header + "\n")
        for i, q in enumerate(QUESTIONS):
            log_block = (
                f"\n[Q{i+1}]: {q}\n" + "-"*40 +
                f"\nBASE MODEL:\n{base_results[i]}\n" + "."*20 +
                f"\nDIRECTRIX MODEL:\n{merged_results[i]}\n" + "-"*40
            )
            print(log_block)
            f.write(log_block + "\n")

    log_event("Audit Complete. All results saved to logs/audit.log")

if __name__ == "__main__":
    run_audit()