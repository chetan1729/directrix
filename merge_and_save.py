# --- 1. HARDWARE LOCK ---
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import keras_hub
import pickle
from datetime import datetime

def log_event(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Merge: {message}")

PRESET_PATH = "/workspace/gemma_keras_preset"
CHECKPOINT_PATH = "checkpoints/ckpt_step_7000.pkl"
EXPORT_PATH = "/workspace/gemma_2b_directrix_final"

def merge_lora():
    log_event("Starting Weight Merger...")
    
    # 1. Load the model with LoRA enabled to match your checkpoint structure
    model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
    model.backbone.enable_lora(rank=8)
    
    log_event(f"Loading weights from {CHECKPOINT_PATH}...")
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    
    # 2. Assign the LoRA weights to the model
    # Note: Using p_vars as identified in your logs
    for v, val in zip(model.trainable_variables, ckpt['p_vars']):
        v.assign(val)
        
    log_event("Performing Surgical Matrix Merger (W = W + A*B)...")
    
    # Map all model variables by path for fast lookup
    vars_by_path = {v.path: v for v in model.variables}
    merged_count = 0

    for path, v in vars_by_path.items():
        if "lora_kernel_a" in path:
            prefix = path.replace("lora_kernel_a", "")
            path_b = prefix + "lora_kernel_b"
            path_base = prefix + "kernel"
            
            if path_b in vars_by_path and path_base in vars_by_path:
                # --- DEFINE VARIABLES FIRST ---
                A = vars_by_path[path].value       # (heads, in, rank)
                B = vars_by_path[path_b].value     # (heads, out)
                W = vars_by_path[path_base].value  # (1, in, out) or (heads, in, out)
                
                # --- MERGE LOGIC ---
                if A.ndim == 3 and B.ndim == 2:
                    # h: heads, i: in_dim, r: rank, o: out_dim
                    delta_w = jnp.einsum('hir,ho->hio', A, B)
                    
                    # Handle the case where base W is (1, in, out) but delta is (8, in, out)
                    target_shape = W.shape
                    if delta_w.shape != target_shape:
                        if target_shape[0] == 1:
                            delta_w = jnp.sum(delta_w, axis=0, keepdims=True)
                        else:
                            delta_w = jnp.reshape(delta_w, target_shape)
                
                elif A.ndim == 2 and B.ndim == 2: # For any dense layers
                    delta_w = jnp.matmul(A, B)
                else:
                    log_event(f"Skipping {path}: Shapes A={A.shape}, B={B.shape}")
                    continue
            
                # Update the base weight
                vars_by_path[path_base].assign(W + delta_w)
                merged_count += 1

    log_event(f"Successfully merged {merged_count} LoRA adapter pairs.")

    # 3. Save as a CLEAN model (No LoRA adapters in the final export)
    log_event("Exporting to standalone preset...")
    final_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
    
    # Sync weights from merged model to clean model
    for v_src in model.backbone.variables:
        if "lora_kernel" not in v_src.path:
            for v_dst in final_model.backbone.variables:
                if v_src.path == v_dst.path:
                    v_dst.assign(v_src.value)
                    break

    final_model.save_to_preset(EXPORT_PATH)
    log_event(f"Final model saved to {EXPORT_PATH}")

if __name__ == "__main__":
    merge_lora()
