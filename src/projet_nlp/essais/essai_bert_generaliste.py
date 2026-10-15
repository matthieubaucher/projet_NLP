from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

def embed(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze()

vec = embed("This stock will explode")
print(vec.shape)
