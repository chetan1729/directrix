# --- 1. HARDWARE LOCK ---
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import gc
import numpy as np
import keras_hub
import pickle
from datetime import datetime
from datasets import load_dataset

def log_event(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open("logs/validation.log", "a") as f:
        f.write(msg + "\n")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# --- 2. INITIALIZE ---
PRESET_PATH = "/workspace/gemma_keras_preset"
CHECKPOINT_PATH = "checkpoints/ckpt_step_7000.pkl"
devices = jax.devices()

log_event(f"Loading model from {PRESET_PATH}...")
model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
model.backbone.enable_lora(rank=8)

# --- 3. UTILITIES ---
def get_batch_logps(logits, labels, mask):
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    mask = mask[:, 1:]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_token_logps = jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    return jnp.sum(per_token_logps * mask, axis=-1)

def preprocess_hh_rlhf(item):
    def split_prompt_response(text):
        parts = text.split("\n\nAssistant: ")
        if len(parts) < 2:
            return text, ""
        prompt = "\n\nAssistant: ".join(parts[:-1]) + "\n\nAssistant: "
        response = parts[-1]
        return prompt, response

    p_chosen, r_chosen = split_prompt_response(item['chosen'])
    p_rejected, r_rejected = split_prompt_response(item['rejected'])
    return {"prompt": p_chosen, "chosen": r_chosen, "rejected": r_rejected}

def tokenize_sample(entry, preprocessor, max_len=512):
    def get_tokens(prompt, response):
        out_p = preprocessor(prompt)
        p_ids = out_p[0] if isinstance(out_p, tuple) else out_p
        p_len = int(jnp.sum(p_ids['padding_mask']))
        out_f = preprocessor(prompt + response)
        f_ids = out_f[0] if isinstance(out_f, tuple) else out_f
        ids = np.array(f_ids['token_ids'])
        mask = np.array(f_ids['padding_mask'], dtype=float)
        mask[:p_len] = 0.0 
        final_ids = np.zeros(max_len, dtype="int32")
        final_mask = np.zeros(max_len, dtype="float32")
        curr_len = min(len(ids), max_len)
        final_ids[:curr_len] = ids[:curr_len]
        final_mask[:curr_len] = mask[:curr_len]
        return final_ids, final_mask

    c_ids, c_mask = get_tokens(entry['prompt'], entry['chosen'])
    r_ids, r_mask = get_tokens(entry['prompt'], entry['rejected'])
    return {"c_ids": c_ids, "c_mask": c_mask, "r_ids": r_ids, "r_mask": r_mask}

# --- 4. LOAD CHECKPOINT ---
if not os.path.exists(CHECKPOINT_PATH):
    log_event(f"ERROR: Checkpoint {CHECKPOINT_PATH} not found.")
    exit(1)

log_event(f"Loading weights from {CHECKPOINT_PATH}...")
with open(CHECKPOINT_PATH, "rb") as f:
    ckpt = pickle.load(f)

p_vars = ckpt['p_vars']
# Reference weights (initial zero LoRA)
initial_p_vars = [jnp.zeros_like(v) for v in p_vars]
base_vars = [v.value for v in model.non_trainable_variables]

# Move to GPU
p_vars = jax.device_put(p_vars, devices[0])
initial_p_vars = jax.device_put(initial_p_vars, devices[0])
base_vars = jax.device_put(base_vars, devices[0])

# --- 5. INFERENCE STEP ---
@jax.jit
def eval_step(p_v, i_p_v, b_v, batch):
    c_in = {"token_ids": batch['c_ids'], "padding_mask": batch['c_ids'] > 0}
    r_in = {"token_ids": batch['r_ids'], "padding_mask": batch['r_ids'] > 0}
    
    # Policy logps
    p_c_logits = model.stateless_call(p_v, b_v, c_in)[0]
    p_r_logits = model.stateless_call(p_v, b_v, r_in)[0]
    p_c_lp = get_batch_logps(p_c_logits, batch['c_ids'], batch['c_mask'])
    p_r_lp = get_batch_logps(p_r_logits, batch['r_ids'], batch['r_mask'])
    
    # Reference logps
    r_c_logits = model.stateless_call(i_p_v, b_v, c_in)[0]
    r_r_logits = model.stateless_call(i_p_v, b_v, r_in)[0]
    r_c_lp = get_batch_logps(r_c_logits, batch['c_ids'], batch['c_mask'])
    r_r_lp = get_batch_logps(r_r_logits, batch['r_ids'], batch['r_mask'])
    
    # Implicit Rewards: log(pi/ref)
    reward_chosen = p_c_lp - r_c_lp
    reward_rejected = p_r_lp - r_r_lp
    
    is_correct = reward_chosen > reward_rejected
    margin = reward_chosen - reward_rejected
    
    return is_correct, margin

# --- 6. VALIDATION LOOP ---
log_event("Loading 'test' split of HH-RLHF...")
test_ds = load_dataset("Anthropic/hh-rlhf", split="test")

log_event("Starting Quantitative Evaluation (1,000 samples)...")
total_correct = 0
total_margin = 0.0
count = 0
NUM_EVAL = 1000

for i, item in enumerate(test_ds):
    if count >= NUM_EVAL:
        break
    
    entry = preprocess_hh_rlhf(item)
    tokens = tokenize_sample(entry, model.preprocessor)
    
    # Add batch dimension
    batch = jax.tree.map(lambda x: jnp.array([x]), tokens)
    batch_gpu = jax.tree.map(lambda x: jax.device_put(x, devices[0]), batch)
    
    is_correct, margin = eval_step(p_vars, initial_p_vars, base_vars, batch_gpu)
    
    total_correct += int(is_correct[0])
    total_margin += float(margin[0])
    count += 1
    
    if count % 100 == 0:
        log_event(f"Processed {count}/{NUM_EVAL} | Running Accuracy: {total_correct/count:.4f}")

final_acc = total_correct / count
avg_margin = total_margin / count

log_event("--- EVALUATION RESULTS ---")
log_event(f"Total Accuracy: {final_acc:.4f}")
log_event(f"Average Reward Margin: {avg_margin:.4f}")
log_event("Validation Complete.")
