import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Check if CUDA is available


# Load model and tokenizer from Hugging Face
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change model if needed
# HF_TOKEN = "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"  # Define your Hugging Face token

# tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, torch_dtype=torch.float16, device_map="auto")
# Define local model path
local_model_path = "Llama-2-7b-chat-hf"  # Replace with the path where you downloaded the model
tokenizer_path = os.path.join(local_model_path)  # Path to tokenizer directory

# Load model and tokenizer from local directory
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.float16, device_map="auto")



@app.get("/")
def home():
    return {"message": "LLaMA 2 API is running!"}

@app.post("/generate")
def generate_text(prompt: str, max_length: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": response}
