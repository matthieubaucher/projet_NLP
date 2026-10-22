import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from projet_nlp.model_sentiment.EmbeddingClassifier import EmbeddingClassifier
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model


# Pour de l'embedding

MPNET= "microsoft/mpnet-base" # Tweets, finance, messages moyens (vitesse moyenne)
ALBERT = "albert-large-v2" # Messages moyens (gros modele, tres lent)
FINBERT_TONE = "yiyanghkust/finbert-tone" # SOTA pour embeddings financiers
ROBERTA = "cardiffnlp/twitter-roberta-base" # compromis vitesse/qualité
ROBERTA_S = "cardiffnlp/twitter-roberta-base-sentiment" # Tweets sentiment                           <- winner
MINI_BERTA = "distilroberta-base" # Tweets, sentiment (vitesse plutot bonne)
GOOGLE = "google/electra-small-discriminator" # Gros datasets, tweets (rapide)
GROS_GOOGLE = "google/electra-base-discriminator" # Sentiment, finance (2x plus lent, comme MPNET)

# PRIVATE
def encode_in_batches(texts, model_name, batch_size=2048):
    model = SentenceTransformer(model_name, device="cuda")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encodage {model_name}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, device="cuda")
        embeddings.append(emb.cpu())  # on sort du GPU pour éviter la saturation
    return torch.cat(embeddings, dim=0)

# PUBLIC
# Ajoute une colonne msg_embedding au df (mais pas inPlace) calculé à partir du modèle demandé
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
def add_message_embedding(df, excluded_df = None, regenerer=False ,model_name=ROBERTA_S, taille_max = 0):
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

    pq.write_table(
        table,
        filename,
        compression="zstd",      
        chunk_size=100_000      
    )

    return df


# PUBLIC
# Ajoute une colonne user_embedding au df (mais pas inPlace) calculé à partir du modèle demandé
# Sauvegarde le résultat. si le calcul a déjà été fait pour ce model_name et cette taille_max et les symboles du modèle
# alors on retoune la précédente sauvegarde sans faire de calculs.
# 
# OUT :
#  - un dataframe de taille "taille_max" ou aussi grand que le précédent si taille_max=0, et contenant en plus
#    les user_embedding
#
# IN :
#  - df : le dataframe de départ
#  - regenerer : si oui alors on refait le calcul et on écrase la sauvegarde
#  - model_name : nom du modèle à utiliser. ex : FINBERT_TONE, MINI_LM, MPNET, ROBERTA, FINBERT
#  - taille_max : pioche ce nombre d'élément au hasard dans df
#
def add_user_embedding(df, regenerer=False ,model_name=ROBERTA_S, taille_max = 0):
    tickers = sorted(df["ticker"].unique())
    tickers_name = "".join(f"_{t}" for t in tickers)
    filename = f"df_with_user_embeddings{tickers_name}_{model_name.replace("/","_")}_{taille_max}.parquet"

    # Ne pas refaire le calcul si déjà fait
    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)

    # Récupération des msg_embedding (nécessaire pour pouvoir construire le user_embedding)
    df_msg_embedding = add_message_embedding(df, excluded_df = None, regenerer=regenerer ,model_name=model_name, taille_max = taille_max)

    # Fabrication du user embedding : il est la moyenne de ses messages du passé
    df_msg_embedding["user_embedding"] = None
    previous_user_id = None
    for idx, row in enumerate(df_msg_embedding.itertuples()):
        user_id = row.user_id
        msg_embedding = row.msg_embedding
        if user_id!=previous_user_id:
            user_emb = np.zeros_like(msg_embedding)           
            nb = 0
            previous_user_id = user_id
        user_emb += msg_embedding
        nb += 1
        df_msg_embedding.loc[idx, "user_embedding"] = user_emb / nb   

    # Sauvegarde
    table = pa.Table.from_pandas(df_msg_embedding, preserve_index=False)
    pq.write_table(
        table,
        filename,
        compression="zstd",      
        chunk_size=100_000      
    )
    return df


# PRIVATE
def create_dataset(df):
    X_train = df["message"].astype(str).to_list()
    Y_train = df["sentiment"].astype(str).map({"BAISSE": 0, "HAUSSE": 1}).to_numpy().astype(int)
    train = list(zip(X_train, Y_train))
    return train

# PRIVATE
def split_size(train):
    train32 = []
    train64 = []
    train128 = []
    train256 = []
    train512 = []

    for text, label in train:
        n_words = len(text.split())

        if n_words <= 32:
            train32.append((text, label))
        elif n_words <= 64:
            train64.append((text, label))
        elif n_words <= 128:
            train128.append((text, label))
        elif n_words <= 256:
            train256.append((text, label))
        else:
            train512.append((text, label))

    return train32, train64, train128, train256, train512



# PUBLIC
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

# PRIVATE
# Entraine un modèle à embeddings sur les sentiments
# Nécessite que au moins certaines lignes soient déjà annotée avec des sentiments pour servir à l'entrainement
#
# OUT : le modèle entrainé
#
# IN : 
#  - model_name : le nom du modèle à embedding à utiliser et à entrainer. 
#  - df : le dataframe pour s'entrainer (tout est utilisé)
#
def train_embedding_model(df, model_name=ROBERTA_S, nom_sauvegarde="dernier", df_validation=None):
    
    # Préparation du train
    df_train = equilibre_HAUSSE_BAISSE(df)
    df_val = equilibre_HAUSSE_BAISSE(df_validation)
    train = create_dataset(df_train)
    validation = create_dataset(df_val)
    
    train32, train64, train128, train256, train512 = split_size(train)
    validation32, validation64, validation128, validation256, validation512 = split_size(validation)

    
    print(f"Nombre d'éléments pour l'entrainement = {len(df_train)}")

    # Création du modèle
    model = EmbeddingClassifier(model_name, nom_sauvegarde).cuda()

    # Entrainement (ne surtout pas prendre plus de 3 époques, ca sur-apprend sinon)
    # epoque 1
    model.fit(train32, epochs=1, lr=4e-5, trainset_validation=validation32, token_size_max = 32)
    model.fit(train64, epochs=1, lr=4e-5, trainset_validation=validation64, token_size_max = 64)
    model.fit(train128, epochs=1, lr=4e-5, trainset_validation=validation128, token_size_max = 128)
    if len(train256)>1:
        if len(validation256)>100:
            val = validation256
        else: 
            val = validation128
        model.fit(train256, epochs=1, lr=4e-5, trainset_validation=val, token_size_max = 256)
    if len(train512)>1:
        if len(validation512)>100:
            val = validation512
        else: 
            val = validation256
        model.fit(train512, epochs=1, lr=4e-5, trainset_validation=val, token_size_max = 512)
    # epoque 2        
    model.fit(train32, epochs=1, lr=2e-5, trainset_validation=validation32, token_size_max = 32)
    model.fit(train64, epochs=1, lr=2e-5, trainset_validation=validation64, token_size_max = 64)        
    model.fit(train128, epochs=1, lr=2e-5, trainset_validation=validation128, token_size_max = 128)
    if len(train256)>1:
        if len(validation256)>100:
            val = validation256
        else: 
            val = validation128
        model.fit(train256, epochs=1, lr=2e-5, trainset_validation=val, token_size_max = 256)
    if len(train512)>1:
        if len(validation512)>100:
            val = validation512
        else: 
            val = validation256
        model.fit(train512, epochs=1, lr=2e-5, trainset_validation=val, token_size_max = 512)


    return model


# PRIVATE
# Utilise un modèle entrainé pour rajouter une colonne generated_sentiment
# IN : le modèle et le dataframe à décorer
# OUT : le dataframe (qui a aussi été modifier inplace), et le score de précision
def add_generated_sentiment(df, model, batch_size=256):
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


# PUBLIC
# Ajoute une colonne generated_sentiment au df (mais pas inPlace) calculé à partir du modèle demandé
# Entraine le modèle si pas déjà fait et le sauvegarde. si le modèle existe déjà alors le charge et l'utilise sans le réentrainer
# 
# OUT :
#  - un dataframe avec une colonne generated_sentiment en plus
#  - le modele entrainé
#
# IN :
#  - df : le dataframe de départ (sans generated_sentiment)
#  - regenerer : si oui alors on recalcule le modèle avant même s'il existait (et on l'écrase ensuite)
#  - model_name : nom du modèle à utiliser. ex : FINBERT_TONE, ROBERTA, ...
#  - taille_max : nombre de tweet d'entrainement du modèle, attention : fait partie du nom de la sauvegarde pour save et load
#  - test : si c'est juste pour du test ou pour remplir la totalite du dataframe
#
def add_message_sentiment(df, regenerer=False ,model_name=ROBERTA_S, taille_max = 400000, test=True):
    tickers = sorted(df["ticker"].unique())
    tickers_name = "".join(f"_{t}" for t in tickers)
    tickers_name += "_test" if test else ""
    filename = f"df_with_generated_sentiment{tickers_name}_{model_name.replace("/","_")}_{taille_max}.parquet"

    if model_name==ROBERTA:
        fn = "grosse-berta"
    elif model_name==MINI_BERTA:
        fn = "micro-berta"
    elif model_name==MPNET:
        fn = "mpnet"
    elif model_name==FINBERT_TONE:
        fn = "finbert"
    elif model_name==ROBERTA_S:
        fn = "petite-berta"
    elif model_name==ALBERT:
        fn = "albert"
    elif model_name==GOOGLE:
        fn = "petit-google"
    elif model_name==GROS_GOOGLE:
        fn = "gros-google"
    else:
        fn = "autre"
    nom_sauvegarde = f"toto_{fn}{taille_max}.pt"

    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        model = load_model(nom_sauvegarde, model_name)
        return pd.read_parquet(filename), model

    df_train = df.sample(n=taille_max, random_state=42)
    df_train, df_test = train_test_split(
        df_train,
        test_size=0.05,      # 4% pour le test
        random_state=42,    
        shuffle=True
    )
    df_test, df_validation = train_test_split(
        df_test,
        test_size=0.2,       # 1% pour la validation inter époque
        random_state=42,    
        shuffle=True
    )

    if os.path.exists(nom_sauvegarde) and not regenerer:   
        print(f"=== Modèle déjà existant : {nom_sauvegarde} ===")
        model = load_model(nom_sauvegarde, model_name)
    else:
        print("TRAIN")

        model = train_embedding_model(df_train, nom_sauvegarde=nom_sauvegarde, df_validation=df_validation, model_name=model_name)
        model.save()

    if test:
        selection = df_test
    else:
        selection = df

    resultat, score = add_generated_sentiment(selection, model)
    print(f"L'annotation a été effectué avec une précision de {score}")

    # Conversion en table Arrow
    table = pa.Table.from_pandas(selection, preserve_index=False)

    pq.write_table(
        table,
        filename,
        compression="zstd",      
        chunk_size=100_000      
    )

    return resultat, model
