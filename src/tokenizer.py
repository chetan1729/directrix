def tokenize_batch(pairs, tokenizer):
    # Log: Tokenizing preference batch
    chosen_ids = tokenizer(pairs['chosen'])
    rejected_ids = tokenizer(pairs['rejected'])
    return {
        "chosen_ids": jnp.array(chosen_ids),
        "rejected_ids": jnp.array(rejected_ids)
    }