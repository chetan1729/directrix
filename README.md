# Directrix: Surgical DPO Alignment for Gemma-2B

Directrix is a research engineering project focused on the alignment of the **Gemma-2B** backbone using **Direct Preference Optimization (DPO)**. By bypassing traditional Reward Modeling, Directrix optimizes the model directly on human preference distributions to achieve high-fidelity HHH (Helpful, Honest, Harmless) compliance.

The project features a custom **Surgical Merge** protocol to integrate LoRA adapters into Grouped Query Attention (GQA) architectures without corrupting sharded tensor dimensions.

## 🚀 Key Achievements
* **Alignment Performance:** Achieved a **62.2% preference accuracy** on the Anthropic HH-RLHF test set.
* **Architecture:** Successfully implemented a vectorized merge for 3D GQA attention kernels using `jax.numpy.einsum`.
* **Efficiency:** Trained on dual **NVIDIA RTX 5090s** using JAX-native sharding and optimization.
* **Robustness:** Verified alignment through a 5-domain "Alignment Tax" audit, maintaining technical STEM proficiency while enhancing safety guardrails.

---

## 🧠 Theoretical Framework

Directrix utilizes the DPO objective to optimize the policy $\pi_{\theta}$ against a frozen reference model $\pi_{ref}$. This eliminates the need for a separate reward model by treating the policy as its own reward estimator.

The optimization objective is defined as:

$$\mathcal{L}_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$



We utilized a KL-divergence constraint of $\beta = 0.1$ to maintain the model's pre-training intelligence while shifting its conversational persona.

---

## 🛠 Technical Innovation: The GQA Surgical Merge

A primary challenge in aligning Gemma-2B is the structural mismatch between LoRA adapters and the Grouped Query Attention (GQA) storage format. Standard weight addition fails due to head-sharding. 

Directrix solves this via a custom **Einstein Summation** merge that reconstructs the attention kernels head-wise before collapsing into the base backbone:



```python
# Surgical merge for sharded MHA/GQA tensors
# h=heads, i=input, r=rank, o=output
delta_w = jnp.einsum('hir,hro->hio', A, B)

# Collapse head dimension for unified backbone storage
if W_base.shape[0] == 1:
    delta_w = jnp.sum(delta_w, axis=0, keepdims=True)

W_final = W_base + delta_w
```

## 📊 Empirical Evaluation

1. Training Convergence & AccuracyThe model was trained for 7,130 steps. The Negative Log Likelihood (NLL) stabilized between 0.6 and 0.8, while Validation Accuracy climbed from a 50% baseline to 62.2%.
2. Reward Margin DistributionThe implicit reward margin (r(x, y_w) - r(x, y_l)) shows a significant rightward shift, with an average margin of 2.3334, proving a statistically significant internalized preference for aligned responses.

## 📂 Project Structure
* src/train.py: JAX-native DPO training loop.

* src/merge_and_save.py: The GQA Surgical Merge logic.

* src/audit.py: Multi-domain capability stress-testing suite.

* logs/: Full training, validation, and showcase logs.

* metrics/: Visualization artifacts for reward distribution and loss analysis.

## 📈 Future Roadmap

* Phase 2: Implement Length-Normalized DPO to mitigate verbosity bias.

* Phase 3: Iterative DPO using Gemini 1.5 Pro as a preference labeler.
