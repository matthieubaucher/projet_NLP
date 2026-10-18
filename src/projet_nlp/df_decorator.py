import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from projet_nlp.model_sentiment.EmbeddingClassifier import EmbeddingClassifier


# Pour de l'embedding

FINBERT_TONE = "yiyanghkust/finbert-tone" # SOTA pour embeddings financiers
ROBERTA = "cardiffnlp/twitter-roberta-base" # compromis vitesse/qualité


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
#  - excluded_df : exclusion de toutes lignes de ce dataframe dans le df pris (utile pour exclure un train et générer un test)
#
def add_message_embedding(df, excluded_df = None, regenerer=False ,model_name=FINBERT_TONE, taille_max = 0):
    tickers = sorted(df["ticker"].unique())
    tickers_name = "".join(f"_{t}" for t in tickers)
    filename = f"df_with_msg_embeddings{tickers_name}_{model_name.replace("/","_")}_{taille_max}.parquet"
    
    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)

    if excluded_df is not None:
        df = df[~df.index.isin(excluded_df.index)]

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

def create_dataset(df):
    X_train = df["message"].astype(str).to_list()
    Y_train = df["sentiment"].astype(str).map({"BAISSE": 0, "HAUSSE": 1}).to_numpy().astype(int)
    train = list(zip(X_train, Y_train))
    return train

# Equilibre le df dataframe pour qu'il y ait autant de hausses que de baisses et pas de unknown
# IN : un df avec des sentiments
# OUT : le df purgé de ses UNKNOWN et d'une partie des HAUSSE ou de ses BAISSE suivant qui était en excés
def equilibre_HAUSSE_BAISSE(df):
    df = df[df["sentiment"]!="UNKNOWN"]
    count_baisse = (df["sentiment"]=="BAISSE").sum()
    count_hausse = (df["sentiment"]=="HAUSSE").sum()
    if count_baisse == count_hausse:
        return df
    df_baisse = df[df["sentiment"] == "BAISSE"]
    df_hausse = df[df["sentiment"] == "HAUSSE"]
    if count_baisse<count_hausse:
        df_hausse_balanced = df_hausse.sample(len(df_baisse), random_state=42)
        df = pd.concat([df_baisse, df_hausse_balanced]).sample(frac=1, random_state=42)
    else:
        df_baisse_balanced = df_baisse.sample(len(df_hausse), random_state=42)
        df = pd.concat([df_hausse, df_baisse_balanced]).sample(frac=1, random_state=42)
    return df

# Entraine un modèle à embeddings sur les sentiments
# Nécessite que au moins certaines lignes soient déjà annotée avec des sentiments pour servir à l'entrainement
#
# OUT : le modèle entrainé
#
# IN : 
#  - model_name : le nom du modèle à embedding à utiliser et à entrainer. 
#  - df : le dataframe pour s'entrainer (tout est utilisé)
#
def train_embedding_model(df, model_name=FINBERT_TONE, nom_sauvegarde="dernier", df_validation=None):
    
    # Préparation du train
    df_train = equilibre_HAUSSE_BAISSE(df)
    df_val = equilibre_HAUSSE_BAISSE(df_validation)
    train = create_dataset(df_train)
    validation = create_dataset(df_val)
    
    print(f"Nombre d'éléments pour l'entrainement = {len(df_train)}")

    # Création du modèle
    model = EmbeddingClassifier(model_name, nom_sauvegarde).cuda()

    # Entrainement (ne surtout pas prendre plus de 3 époques, ca sur-apprend sinon)
    model.entraine(train, epochs=1, epochfinetune=1, lr=8e-5, trainset_validation=validation)
    model.entraine(train, epochs=0, epochfinetune=2, lr=5e-5, trainset_validation=validation)

    return model


# Utilise un modèle entrainé pour rajouter une colonne generated_sentiment
# IN : le modèle et le dataframe à décorer
# OUT : le dataframe (qui a aussi été modifier inplace), et le score de précision
def add_generated_sentiment(df, model, batch_size=512):
    print("Inférence")

    sentences = df["message"].astype(str).to_list()
   
    sentiments =[]
    for i in tqdm(range(0, len(sentences), batch_size), desc="Inférence"):
        batch = sentences[i:i+batch_size]
        sentiments_pack = model.predict(batch)   
        sentiments.extend(sentiments_pack)
    df["generated_sentiment"] = sentiments

    df_tested = df[df["sentiment"]!="UNKNOWN"]
    ok = (df_tested["sentiment"] == df_tested["generated_sentiment"]).sum()
    nb = len(df_tested)
    score = 100 * ok / nb

    return df, score
