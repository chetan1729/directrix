import os
import datetime
import jax
import jax.numpy as jnp
import optax
import gc
import numpy as np

# --- 1. HARDWARE & BACKEND LOCK ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import keras
import keras_hub

def log_event(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# --- 2. INITIALIZE MODELS ---
PRESET_PATH = "/workspace/gemma_keras_preset"
devices = jax.devices()

log_event("Loading models into system memory...")
model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)
ref_model = keras_hub.models.GemmaCausalLM.from_preset(PRESET_PATH)

# --- 3. DPO UTILITIES ---
def get_batch_logps(logits, labels):
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return jnp.sum(log_probs * one_hot, axis=-1).sum(axis=-1)

def dpo_loss(p_c_lp, p_r_lp, r_c_lp, r_r_lp, beta=0.1):
    logits = beta * ((p_c_lp - r_c_lp) - (p_r_lp - r_r_lp))
    return -jax.nn.log_sigmoid(logits).mean(), jnp.mean((p_c_lp - r_c_lp) > (p_r_lp - r_r_lp))

# --- 4. DATA PREP ---
def tokenize_batch(pairs, preprocessor, max_len=1024):
    def get_raw_ids(text_list):
        out = preprocessor(text_list)
        
        # 1. Standardize to a list of sequences
        # If it's a dict, it's a batch of sequences. If it's a list, it's a list of dicts.
        if isinstance(out, dict):
            sequences = out["token_ids"]
        elif isinstance(out, (list, tuple)) and isinstance(out[0], dict):
            sequences = [item["token_ids"] for item in out]
        else:
            sequences = out

        # 2. Create the clean matrix
        batch_size = len(text_list)
        final_matrix = np.zeros((batch_size, max_len), dtype="int32")
        
        # 3. Flatten and slot in each sequence
        for i, seq in enumerate(sequences):
            # Force conversion to raw numpy, then flatten to 1D, then to list
            tokens = np.array(seq).flatten().astype(int).tolist()
            length = min(len(tokens), max_len)
            final_matrix[i, :length] = tokens[:length]
            
        return final_matrix

    return {
        "c_ids": jnp.array(get_raw_ids(pairs['chosen'])), 
        "r_ids": jnp.array(get_raw_ids(pairs['rejected']))
    }

# --- 5. SURGICAL SHARDING ---
log_event("Applying Mixed-Precision Sharding...")

# We MUST keep the variables as a dictionary (tree) for stateless_call
p_vars = {v.path: jax.device_put(v.value, devices[0]) for v in model.trainable_variables}
r_vars = {v.path: jax.device_put(v.value.astype(jnp.bfloat16), devices[1]) for v in ref_model.variables}

# Initialize Optimizer (Optax handles dicts/trees perfectly)
optimizer = optax.adamw(learning_rate=4e-7, weight_decay=0.01)

del ref_model
gc.collect()
jax.clear_caches()

log_event("Initializing optimizer states (Headroom created)...")
opt_state = optimizer.init(p_vars)

# --- 6. TRAINING STEP (Dictionary-Aware) ---
@jax.jit
def train_step(p_v, r_v, o_s, batch):
    def compute_loss(params):
        p_in = {"token_ids": batch['c_ids'], "padding_mask": jnp.ones_like(batch['c_ids'], dtype=bool)}
        r_in = {"token_ids": batch['r_ids'], "padding_mask": jnp.ones_like(batch['r_ids'], dtype=bool)}

        # Passing params as a DICT satisfies the Gemma forward pass structure
        p_c_logits = model.stateless_call(params, [], p_in)
        p_r_logits = model.stateless_call(params, [], r_in)
        
        r_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), r_v)
        r_c_logits = model.stateless_call(r_f32, [], p_in)
        r_r_logits = model.stateless_call(r_f32, [], r_in)
        
        loss, reward = dpo_loss(
            get_batch_logps(p_c_logits, batch['c_ids']),
            get_batch_logps(p_r_logits, batch['r_ids']),
            get_batch_logps(r_c_logits, batch['c_ids']),
            get_batch_logps(r_r_logits, batch['r_ids'])
        )
        return loss, reward

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, rew), grads = grad_fn(p_v)
    updates, next_o_s = optimizer.update(grads, o_s, p_v)
    return optax.apply_updates(p_v, updates), next_o_s, loss, rew

# --- 7. RUN ---
raw_data = {
    "chosen": ["The RTX 5090 uses Blackwell architecture for 32GB VRAM."],
    "rejected": ["I am an AI and cannot discuss hardware specifics in detail."]
}

log_event("Forging Clean Tensors...")
tokenized_batch = tokenize_batch(raw_data, model.preprocessor)
tokenized_batch = jax.tree.map(lambda x: jax.device_put(x, devices[0]), tokenized_batch)

log_event("Starting DPO Loop (Blackwell Compilation Incoming)...")
# Note: Always add log putting.
for step in range(1, 51):
    try:
        p_vars, opt_state, l_val, r_val = train_step(p_vars, r_vars, opt_state, tokenized_batch)
        print(f"Step {step} | Loss: {float(l_val):.4f} | Reward: {float(r_val):.4f}")
    except Exception as e:
        log_event(f"❌ Crash at Step {step}: {e}")
        break
        
# Save
for v, val in zip(model.trainable_variables, p_vars): v.assign(val)
model.save_weights("/root/dpo_gemma/checkpoints/gemma_dpo_final.weights.h5")
log_event("✅ Sprint complete.")