import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm

def encode_in_batches(texts, model_name, batch_size=2048):
    model = SentenceTransformer(model_name, device="cuda")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encodage {model_name}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, device="cuda")
        embeddings.append(emb.cpu())  # on sort du GPU pour éviter la saturation
    return torch.cat(embeddings, dim=0)

def add_message_embedding(df, regenerer=False ,model_name="yiyanghkust/finbert-tone"):
    filename = f"df_with_msg_embeddings_{model_name.replace("/","_")}.parquet"
    
    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)
    
    embeddings = encode_in_batches(df["message"].tolist(), model_name)

    df["msg_embedding"] = [emb.numpy() for emb in embeddings]

    print("=== Sauvegarde df avec message embeddings ===")
    df.to_parquet(filename, compression="zstd")
    return df
