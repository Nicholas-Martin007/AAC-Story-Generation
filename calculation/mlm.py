from transformers import BertForMaskedLM
import torch

# Load model (from Hugging Face or local safetensor)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Get all parameters
for name, param in model.named_parameters():
    print(f'Name: {name}, Shape: {param.shape}')
    # If you want, you can inspect values (be careful, large!)
    print(param.data)  # tensor of weights
    break  # remove break if you want to iterate through all
