from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-2b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Model loaded successfully!")

prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt")

print("Generating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=20
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
