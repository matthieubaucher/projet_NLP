from sentence_transformers import SentenceTransformer
import torch

# construction du model pour transformer les tweets en embeddings

model = SentenceTransformer("yiyanghkust/finbert-tone", device="cuda")

print(torch.__version__) 
print("Device utilisé :", model.device)


# petit esssai ...

embedding = model.encode("This stock will explode")
print(embedding.shape)