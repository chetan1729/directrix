# --- 1. HARDWARE LOCK ---
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import gc
import numpy as np
import keras_hub
import time
import pickle
from datetime import datetime
from datasets import load_dataset

def log_event(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open("logs/training.log", "a") as f:
        f.write(msg + "\n")

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# --- 2. INITIALIZE ---
PRESET_PATH = "/workspace/gemma_keras_preset"
devices = jax.devices()

log_event("Loading models...")
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

def dpo_loss(p_c_lp, p_r_lp, r_c_lp, r_r_lp, beta=0.1):
    logits = beta * ((p_c_lp - r_c_lp) - (p_r_lp - r_r_lp))
    loss = -jax.nn.log_sigmoid(logits).mean()
    reward_acc = jnp.mean((p_c_lp - r_c_lp) > (p_r_lp - r_r_lp))
    return loss, reward_acc

# --- 4. DATA PIPELINE (HH-RLHF) ---
def preprocess_hh_rlhf(item):
    """Splits on the last 'Assistant:' tag to separate prompt from response."""
    def split_prompt_response(text):
        parts = text.split("\n\nAssistant: ")
        if len(parts) < 2:
            return text, ""
        prompt = "\n\nAssistant: ".join(parts[:-1]) + "\n\nAssistant: "
        response = parts[-1]
        return prompt, response

    p_chosen, r_chosen = split_prompt_response(item['chosen'])
    p_rejected, r_rejected = split_prompt_response(item['rejected'])
    
    return {
        "prompt": p_chosen,
        "chosen": r_chosen,
        "rejected": r_rejected
    }

def tokenize_and_cache(dataset, preprocessor, max_len=512, cache_path="dataset_cache.pkl"):
    if os.path.exists(cache_path):
        log_event(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log_event("Processing and tokenizing HH-RLHF dataset...")
    processed_data = []
    # Using a subset for speed in this setup, or full if memory allows
    # For 10,000 steps, we need enough data.
    for i, item in enumerate(dataset):
        if i >= 20000: break # Load enough for variety
        entry = preprocess_hh_rlhf(item)
        
        # Tokenization logic
        def get_tokens(prompt, response):
            out_p = preprocessor(prompt)
            p_ids = out_p[0] if isinstance(out_p, tuple) else out_p
            p_len = int(jnp.sum(p_ids['padding_mask']))
            
            out_f = preprocessor(prompt + response)
            f_ids = out_f[0] if isinstance(out_f, tuple) else out_f
            
            ids = np.array(f_ids['token_ids'])
            mask = np.array(f_ids['padding_mask'], dtype=float)
            mask[:p_len] = 0.0 # Mask prompt
            
            final_ids = np.zeros(max_len, dtype="int32")
            final_mask = np.zeros(max_len, dtype="float32")
            curr_len = min(len(ids), max_len)
            final_ids[:curr_len] = ids[:curr_len]
            final_mask[:curr_len] = mask[:curr_len]
            return final_ids, final_mask

        c_ids, c_mask = get_tokens(entry['prompt'], entry['chosen'])
        r_ids, r_mask = get_tokens(entry['prompt'], entry['rejected'])
        
        processed_data.append({
            "c_ids": c_ids, "c_mask": c_mask,
            "r_ids": r_ids, "r_mask": r_mask
        })
        if i % 1000 == 0: print(f"Tokenized {i} items...")

    log_event(f"Caching dataset to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(processed_data, f)
    return processed_data

class ShuffledIterator:
    def __init__(self, data, batch_size=4):
        self.data = data
        self.batch_size = batch_size
        self.indices = np.arange(len(data))
        self.pos = 0
        np.random.shuffle(self.indices)

    def __next__(self):
        if self.pos + self.batch_size > len(self.indices):
            np.random.shuffle(self.indices)
            self.pos = 0
        
        batch_idx = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        
        batch = [self.data[i] for i in batch_idx]
        return {
            "c_ids": jnp.array([x['c_ids'] for x in batch]),
            "c_mask": jnp.array([x['c_mask'] for x in batch]),
            "r_ids": jnp.array([x['r_ids'] for x in batch]),
            "r_mask": jnp.array([x['r_mask'] for x in batch]),
        }

# Load and prepare
raw_ds = load_dataset("Anthropic/hh-rlhf", split="train")
tokenized_data = tokenize_and_cache(raw_ds, model.preprocessor)
train_iter = ShuffledIterator(tokenized_data, batch_size=2)

# --- 5. VARIABLE EXTRACTION ---
p_vars = [v.value for v in model.trainable_variables]
initial_p_vars = [jnp.array(v.value) for v in model.trainable_variables]
base_vars = [v.value for v in model.non_trainable_variables]

p_vars = jax.device_put(p_vars, devices[0])
initial_p_vars = jax.device_put(initial_p_vars, devices[0])
base_vars = jax.device_put(base_vars, devices[0])

optimizer = optax.adamw(learning_rate=5e-6)
opt_state = optimizer.init(p_vars)

# --- 6. TRAINING STEP ---
@jax.jit
def train_step(p_v, i_p_v, b_v, o_s, batch):
    def compute_loss(params):
        c_in = {"token_ids": batch['c_ids'], "padding_mask": batch['c_ids'] > 0}
        r_in = {"token_ids": batch['r_ids'], "padding_mask": batch['r_ids'] > 0}
        p_c_logits = model.stateless_call(params, b_v, c_in)[0]
        p_r_logits = model.stateless_call(params, b_v, r_in)[0]
        r_c_logits = model.stateless_call(i_p_v, b_v, c_in)[0]
        r_r_logits = model.stateless_call(i_p_v, b_v, r_in)[0]
        loss, reward = dpo_loss(
            get_batch_logps(p_c_logits, batch['c_ids'], batch['c_mask']),
            get_batch_logps(p_r_logits, batch['r_ids'], batch['r_mask']),
            get_batch_logps(r_c_logits, batch['c_ids'], batch['c_mask']),
            get_batch_logps(r_r_logits, batch['r_ids'], batch['r_mask'])
        )
        return loss, reward

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, rew), grads = grad_fn(p_v)
    updates, next_o_s = optimizer.update(grads, o_s, p_v)
    return optax.apply_updates(p_v, updates), next_o_s, loss, rew

# --- 7. RUN ---
log_event("Starting 10,000 Step DPO Run...")
TOTAL_STEPS = 10000
CHECKPOINT_INTERVAL = 500

for step in range(1, TOTAL_STEPS + 1):
    batch = next(train_iter)
    batch_gpu = jax.tree.map(lambda x: jax.device_put(x, devices[0]), batch)
    
    p_vars, opt_state, l_val, r_val = train_step(p_vars, initial_p_vars, base_vars, opt_state, batch_gpu)
    
    if step % 10 == 0:
        log_event(f"Step {step:05d} | Loss: {float(l_val):.6f} | Reward Acc: {float(r_val):.4f}")
    
    if step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = f"checkpoints/ckpt_step_{step}.pkl"
        log_event(f"Saving checkpoint to {ckpt_path}...")
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "p_vars": p_vars,
                "opt_state": opt_state,
                "step": step
            }, f)

log_event("Directrix Complete.")
