import jax
import jax.numpy as jnp

def get_batch_logps(logits, labels, average_log_prob=False):
    """
    Calculates the log-probabilities of the given labels under the given logits.
    """
    # Shift logits and labels to align for next-token prediction
    # (Standard for Causal LM: token at i predicts token at i+1)
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]

    # Log-softmax over the vocabulary dimension
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Extract the log-probs for the actual token IDs present in 'labels'
    # We use jnp.take_along_axis to gather the specific logits
    per_token_logps = jnp.take_along_axis(
        log_probs, labels[..., None], axis=-1
    ).squeeze(-1)

    if average_log_prob:
        return per_token_logps.mean(axis=-1)
    else:
        # DPO typically uses the sum of log-probs across the sequence
        return per_token_logps.sum(axis=-1)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             reference_chosen_logps, reference_rejected_logps, 
             beta=0.1):
    """
    The DPO objective function.
    """
    # Calculate the log-ratio between policy and reference
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # The DPO loss formula: -log_sigmoid(beta * (chosen_ratio - rejected_ratio))
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -jax.nn.log_sigmoid(logits).mean()
    
    # Log putting: track implicit reward
    reward_accuracy = jnp.mean(chosen_logratios > rejected_logratios)
    
    return loss, reward_accuracy