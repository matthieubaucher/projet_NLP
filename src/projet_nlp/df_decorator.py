import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from projet_nlp.model_sentiment.EmbeddingClassifier import EmbeddingClassifier
from projet_nlp.model_sentiment.EmbeddingClassifierCtx import EmbeddingClassifierCtx
from projet_nlp.model_sentiment.EmbeddingClassifierCtx2 import EmbeddingClassifierCtx2
from projet_nlp.model_sentiment.EmbeddingClassifierCtx3 import EmbeddingClassifierCtx3
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model as load_model_0
from projet_nlp.model_sentiment.EmbeddingClassifierCtx import load_model as load_model_1
from projet_nlp.model_sentiment.EmbeddingClassifierCtx2 import load_model as load_model_2
from projet_nlp.model_sentiment.EmbeddingClassifierCtx3 import load_model as load_model_3
from projet_nlp.df_cleaner import filter_bad_phrases
from multiprocessing import Process, Event


# Pour de l'embedding

MPNET= "microsoft/mpnet-base" # Tweets, finance, messages moyens (vitesse moyenne)
ALBERT = "albert-large-v2" # Messages moyens (gros modele, tres lent)
FINBERT_TONE = "yiyanghkust/finbert-tone" # SOTA pour embeddings financiers
ROBERTA = "cardiffnlp/twitter-roberta-base" # compromis vitesse/qualité
ROBERTA_S = "cardiffnlp/twitter-roberta-base-sentiment" # Tweets sentiment                           <- winner
MINI_BERTA = "distilroberta-base" # Tweets, sentiment (vitesse plutot bonne)
GOOGLE = "google/electra-small-discriminator" # Gros datasets, tweets (rapide)
GROS_GOOGLE = "google/electra-base-discriminator" # Sentiment, finance (2x plus lent, comme MPNET)


def save_async(obj, filename, save_done):
    def _save(o, f):
        torch.save(o, f)
        save_done.set()
    save_done.clear()
    p = Process(target=_save, args=(obj, filename))
    p.start()  # pas daemon → survit au parent


def encode_in_batches(texts, model_name, batch_size=1024, embeddings = []):
    save_done = Event()
    save_done.set() 
    model = SentenceTransformer(model_name, device="cuda")
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encodage {model_name}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, device="cuda")
        embeddings.extend(emb.cpu())  # on sort du GPU pour éviter la saturation
        if i%(500*batch_size)==0:
            save_done.wait()  # attend si une sauvegarde précédente n'a pas fini
            save_async(embeddings.copy(), "parcour.tmp", save_done)
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
#  - taille_max : pioche ce nombre d'élément au hasard dans df, si 0 : ne mélange rien (important !) et conserve tout
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
        df = df.copy() # pas de mélange et on garde tout


    df = df[df['message'].apply(lambda x: x.isascii())]


    msg = df["message"].tolist()

    if os.path.exists("parcour.tmp"):
        print("le fichier parcour existe => reprise")
        embeddings_tmp = torch.load("parcour.tmp")
        nb = len(embeddings_tmp)
        msg = msg[nb:]
        print(f"reprise du parcour avec déjà {nb} embeddings stockés")
    else:
        nb = 0
        embeddings_tmp = []

    if nb < 1500000:
        embeddings = encode_in_batches(msg, model_name,embeddings=embeddings_tmp)
        nb = len(embeddings)
    else:
        # tant pis on s'en contentera ...
        embeddings = embeddings_tmp
    embeddings = embeddings[nb-1500000:nb]
    df = df.iloc[nb-1500000:nb]

    print(f"conversion en float")
    df["msg_embedding"] = [emb.numpy().astype("float32") for emb in embeddings]
    #df["msg_embedding"] = [emb.numpy().astype("float32").tolist() for emb in embeddings]

    print("Conversion en table Arrow")
    table = pa.Table.from_pandas(df, preserve_index=False)

    print(f"sauvegarde {filename}")
    pq.write_table(
        table,
        filename,
        compression="zstd",      
        chunk_size=100_000      
    )

    print(f"sauvegarde {filename} ok")
    os.remove("parcour.tmp")
    return df


# PUBLIC
# Ajoute une colonne user_embedding au df (mais pas inPlace) calculé à partir du modèle demandé
# Sauvegarde le résultat. si le calcul a déjà été fait pour ce model_name et cette taille_max et les symboles du modèle
# alors on retoune la précédente sauvegarde sans faire de calculs.
# 
# OUT :
#  - un dataframe de taille "taille_max" ou aussi grand que le précédent si taille_max=0, et contenant en plus
#    les user_embedding  TODO : bon finalement j'équilibre le dataset HAUSSE/BAISSE et je vire les UNKNOWN
#
# IN :
#  - df : le dataframe de départ
#  - regenerer : si oui alors on refait le calcul et on écrase la sauvegarde
#  - model_name : nom du modèle à utiliser. ex : FINBERT_TONE, MINI_LM, MPNET, ROBERTA, FINBERT
#
def add_user_embedding(df, regenerer=False, model_name=ROBERTA_S):
    tickers = sorted(df["ticker"].unique())
    tickers_name = "".join(f"_{t}" for t in tickers)
    filename = f"df_with_user_embeddings{tickers_name}_{model_name.replace('/', '_')}.parquet"

    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)
    else:
        print(f"Recréation de {filename} => regénération")

    # Embeddings des messages
    df_msg = equilibre_HAUSSE_BAISSE(df)
    df_msg = add_message_embedding(df_msg, excluded_df=None, regenerer=True, 
                                   model_name=model_name, taille_max=0)
    print("Messages embedding chargés")

    # Tri strictement temporel
    df_msg = df_msg.sort_values(["user_id", "day", "message_id"]).reset_index(drop=True)
    print("Tri temporel ok")

    # Nouvelle colonne
    df_msg["user_embedding"] = None
    df_msg["user_previous"] = None

    # Calcul cumulatif par user
    for user_id, group in df_msg.groupby("user_id"):
        cumul = np.zeros_like(group.iloc[0].msg_embedding)
        emb_by_ticker = {}
        previous_by_ticker = {}
        nb = 0

        for idx, row in group.iterrows():
            # Embedding = moyenne des messages précédents
            if nb == 0:
                df_msg.at[idx, "user_embedding"] = cumul.copy()
            else:
                df_msg.at[idx, "user_embedding"] = (cumul / nb).copy()

            ticker = df_msg.at[idx, "ticker"]
            previous = previous_by_ticker.get(ticker)
            if previous is None:
                previous = row.msg_embedding.copy()
            else:
                previous = previous.copy()
            emb = emb_by_ticker.get(ticker)
            if emb is None:
                emb = previous
            else:
                emb = emb.copy()
            previous_by_ticker[ticker] = row.msg_embedding.copy()
            emb = (emb+previous) / 2
            emb_by_ticker[ticker] = emb.copy()
            df_msg.at[idx, "user_previous"] = emb.copy()

            # Mise à jour du cumul
            cumul += row.msg_embedding
            nb += 1
    print("cumul ok")

    # Sauvegarde
    table = pa.Table.from_pandas(df_msg, preserve_index=False)
    pq.write_table(table, filename, compression="zstd", chunk_size=100_000)
    print("Sauvegarde ok")

    return df_msg

# PRIVATE
def create_dataset(df, withCtx=0):
    print("df len =", len(df))
    X_train = df["message"].astype(str).to_list()
    if withCtx==1 or withCtx==2:
        X_train_ctx = df["user_embedding"].to_list()
    if withCtx==2 or withCtx==3:
        X_train_ctx2 = df["user_previous"].to_list()
    Y_train = df["sentiment"].astype(str).map({"BAISSE": 0, "HAUSSE": 1}).to_numpy().astype(int)
    if withCtx==3:
        train = list(zip(X_train, X_train_ctx2, Y_train))
    elif withCtx==1:
        train = list(zip(X_train, X_train_ctx, Y_train))
    elif withCtx==0:
        train = list(zip(X_train, Y_train))
    else:
        train = list(zip(X_train, X_train_ctx, X_train_ctx2, Y_train))
    print("train len =", len(train))
    return train

# PRIVATE
def split_size(train, withCtx):
    train32 = []
    train64 = []
    train128 = []
    train256 = []
    train512 = []

    if withCtx==3:
        for text, user_prev, label in train:
            n_words = len(text.split())

            if n_words <= 32:
                train32.append((text, user_prev, label))
            elif n_words <= 64:
                train64.append((text, user_prev, label))
            elif n_words <= 128:
                train128.append((text, user_prev, label))
            elif n_words <= 256:
                train256.append((text, user_prev, label))
            else:
                train512.append((text, user_prev, label))
    elif withCtx==2:
        for text, user_emb, user_prev, label in train:
            n_words = len(text.split())

            if n_words <= 32:
                train32.append((text, user_emb, user_prev, label))
            elif n_words <= 64:
                train64.append((text, user_emb, user_prev, label))
            elif n_words <= 128:
                train128.append((text, user_emb, user_prev, label))
            elif n_words <= 256:
                train256.append((text, user_emb, user_prev, label))
            else:
                train512.append((text, user_emb, user_prev, label))
    elif withCtx==1:
        for text, user_emb, label in train:
            n_words = len(text.split())

            if n_words <= 32:
                train32.append((text, user_emb,label))
            elif n_words <= 64:
                train64.append((text, user_emb, label))
            elif n_words <= 128:
                train128.append((text, user_emb, label))
            elif n_words <= 256:
                train256.append((text, user_emb, label))
            else:
                train512.append((text, user_emb, label))
    else:
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
    #print("toto debut=",len(df))
    df = df[df["sentiment"]!="UNKNOWN"]
    count_baisse = (df["sentiment"]=="BAISSE").sum()
    count_hausse = (df["sentiment"]=="HAUSSE").sum()
    #print("toto count_baisse=",count_baisse)
    #print("toto count_hausse=",count_hausse)
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
    #print("toto=",len(df))
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
def train_embedding_model(df, model_name=ROBERTA_S, nom_sauvegarde="dernier", df_validation=None, withCtx=0):
    
    # Préparation du train
    df_train = equilibre_HAUSSE_BAISSE(df)
    df_val = equilibre_HAUSSE_BAISSE(df_validation)
    train = create_dataset(df_train, withCtx)
    validation = create_dataset(df_val, withCtx)
    #print("taille train =",len(train))
    #print("taille validation =",len(validation))
    train32, train64, train128, train256, train512 = split_size(train, withCtx)
    validation32, validation64, validation128, validation256, validation512 = split_size(validation, withCtx)
    #print("taille validation32 =",len(validation32))
    #print("taille validation64 =",len(validation64))
    #print("taille train32 =",len(train32))
    #print("taille train64 =",len(train64))
    
    print(f"Nombre d'éléments pour l'entrainement = {len(df_train)}")

    # Création du modèle
    if withCtx==3:
        model = EmbeddingClassifierCtx3(model_name, nom_sauvegarde).cuda()
    elif withCtx==2:
        model = EmbeddingClassifierCtx2(model_name, nom_sauvegarde).cuda()
    elif withCtx==1:
        model = EmbeddingClassifierCtx(model_name, nom_sauvegarde).cuda()
    else:
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
def add_generated_sentiment(df, model, batch_size=256, withCtx=0):
    print("Inférence")

    sentences = df["message"].astype(str).to_list()
    if withCtx==1 or withCtx==2:
        user_embeddings = df["user_embedding"].to_list()
    if withCtx==2 or withCtx==3:
        user_previous = df["user_previous"].to_list()

    sentiments =[]
    for i in tqdm(range(0, len(sentences), batch_size), desc="Inférence"):
        batch = sentences[i:i+batch_size]
        if withCtx==3:
            batch_ctx2 = user_previous[i:i+batch_size]
            sentiments_pack = model.predict(batch, batch_ctx2)             
        elif withCtx==2:
            batch_ctx = user_embeddings[i:i+batch_size]
            batch_ctx2 = user_previous[i:i+batch_size]
            sentiments_pack = model.predict(batch, batch_ctx, batch_ctx2)             
        elif withCtx==1:
            batch_ctx = user_embeddings[i:i+batch_size]
            sentiments_pack = model.predict(batch, batch_ctx) 
        else:    
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
def add_message_sentiment(df, regenerer=False ,model_name=ROBERTA_S, taille_max = 400000, test=True, withCtx=0):
    tickers = sorted(df["ticker"].unique())
    tickers_name = "".join(f"_{t}" for t in tickers)
    tickers_name += "_test" if test else ""
    tickers_name += f"_withCtx{withCtx}"
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
    nom_sauvegarde = f"toto_{fn}{taille_max}_ctx{withCtx}.pt"

    if os.path.exists(filename) and not regenerer:
        print(f"=== Fichier déjà existant : {filename} ===")
        if withCtx==3:
            model = load_model_3(nom_sauvegarde, model_name)        
        elif withCtx==2:
            model = load_model_2(nom_sauvegarde, model_name)
        elif withCtx==1:
            model = load_model_1(nom_sauvegarde, model_name)
        else:
            model = load_model_0(nom_sauvegarde, model_name)
        return pd.read_parquet(filename), model
    print("taille_max=",taille_max)

    df_train = df.sample(n=taille_max, random_state=42)
    print("size df_train=",len(df_train))
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
    print("size df_train=",len(df_train))
    print("size df_test=",len(df_test))
    print("size df_validation=",len(df_validation))

    if os.path.exists(nom_sauvegarde) and not regenerer:   
        print(f"=== Modèle déjà existant : {nom_sauvegarde} ===")
        if withCtx==3:
            model = load_model_3(nom_sauvegarde, model_name)        
        elif withCtx==2:
            model = load_model_2(nom_sauvegarde, model_name)
        elif withCtx==1:
            model = load_model_1(nom_sauvegarde, model_name)
        else:
            model = load_model_0(nom_sauvegarde, model_name)
    else:
        print(f"TRAIN withCtx={withCtx}")
        model = train_embedding_model(df_train, nom_sauvegarde=nom_sauvegarde, df_validation=df_validation, model_name=model_name, withCtx=withCtx)
        model.save()

    if test:
        selection = df_test
    else:
        selection = df

    resultat, score = add_generated_sentiment(selection, model, withCtx=withCtx)
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
