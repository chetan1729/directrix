# Validation Inference
print("\n--- POST-ALIGNMENT TEST ---")
test_prompt = "What hardware are we using today?"
# Call model using the NEWly trained variables
output = model.generate(test_prompt, max_length=64, transformer_vars=policy_vars)
print(f"Model Output: {output}")