import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


FINBERT_TONE = "yiyanghkust/finbert-tone" # SOTA pour embeddings financiers
MINI_LM = "sentence-transformers/all-MiniLM-L12-v2" # modèle plus petit et rapide, embeddings 384
MPNET = "sentence-transformers/all-mpnet-base-v2" # tres bon modèle généraliste
ROBERTA = "cardiffnlp/twitter-roberta-base" # compromis vitesse/qualité
FINBERT = "ProsusAI/finbert"  # mieux pour sentiments, moins bien pour embeddings


def encode_in_batches(texts, model_name, batch_size=2048):
    model = SentenceTransformer(model_name, device="cuda")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encodage {model_name}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, device="cuda")
        embeddings.append(emb.cpu())  # on sort du GPU pour éviter la saturation
    return torch.cat(embeddings, dim=0)


# Ajoute une colonne embedding au df (mais pas inPlace) calculé à partir du modèle demandé
# Sauvegarde le résultat. si le calcul a déjà été fait pour ce model_name et cette taille_max et les symboles du modèle
# alors on retoune la précédente sauvegarde sans faire de calculs.
# 
# OUT :
#  - un dataframe de taille "taille_max" ou aussi grand que le précédent si taille_max=0, et contenant en plus
#    les embeddings
#
# IN :
#  - df : le dataframe de départ
#  - regenerer : si oui alors on refait le calcul et on écrase la sauvegarde
#  - model_name : nom du modèle à utiliser. ex : FINBERT_TONE, MINI_LM, MPNET, ROBERTA, FINBERT
#  - taille_max : pioche ce nombre d'élément au hasard dans df
#
def add_message_embedding(df, regenerer=False ,model_name=FINBERT_TONE, taille_max = 0):
    tickers = sorted(df["ticker"].unique())
    tickers_name = "".join(f"_{t}" for t in tickers)
    filename = f"df_with_msg_embeddings{tickers_name}_{model_name.replace("/","_")}_{taille_max}.parquet"
    
    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)

    if taille_max>0:
        print(f"Creation d'un df de taille {taille_max} sur {len(df)} existants")
        df = df.sample(n=taille_max, random_state=42)
    else:
        df = df.copy()
    
    embeddings = encode_in_batches(df["message"].tolist(), model_name)

    df["msg_embedding"] = [emb.numpy().astype("float32") for emb in embeddings]

    # Conversion en table Arrow
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Écriture par chunks
    pq.write_table(
        table,
        filename,
        compression="zstd",      # tu peux mettre snappy si tu veux
        chunk_size=100_000       # ici le chunk est réellement utilisé
    )

    return df
