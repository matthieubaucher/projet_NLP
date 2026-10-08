import pandas as pd
import numpy as np
import glob
import os
import re
import yfinance as yf
from multiprocessing import Pool
import time
import hashlib

DATA_PATH = "./data/"
SENTIMENT_PATH = DATA_PATH + "sentiments/"
MESSAGE_PATH = DATA_PATH + "messages/"

START_PERIOD = "2008-01-01" 
END_PERIOD = "2023-01-31"

START_PERIOD_PANDA = pd.to_datetime(START_PERIOD)

# ============================================================
# 1. Utilitaires
# ============================================================

def parse_symbols(x):
    """Convertit symbol_list en vraie liste Python."""
    if isinstance(x, str):
        return eval(x) if x.startswith("[") else [x]
    return x

def clean_text(t):
    """Nettoyage du texte."""
    if pd.isna(t):
        t = ""
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ============================================================
# 2. Lecture + filtrage des sentiments (streaming)
# ============================================================

def _load_one_sentiment(args):
    path, symbols = args

    try:
        s = pd.read_csv(
            path,
            usecols=["message_id", "user_id", "created_at", "sentiment", "symbol_list"],
            dtype={"message_id": "int64"}
        )

        s["symbol_list"] = s["symbol_list"].apply(parse_symbols)

        # filtrage par symboles
        s = s[s["symbol_list"].apply(lambda lst: any(sym in symbols for sym in lst))]
        print(f"[sentiment] Lecture : {path} ok")
        return s

    except Exception as e:
        print(f"[ERREUR] {path} : {e}")
        return pd.DataFrame()

def load_filtered_sentiments(symbols):
    filename = f"dfs_{'_'.join(symbols)}.parquet"

    if os.path.exists(filename):
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)

    paths = glob.glob(SENTIMENT_PATH + "sentiment_*.csv")

    with Pool() as pool:
        dfs = pool.map(_load_one_sentiment, [(p, symbols) for p in paths])

    if len(dfs) == 0:
        return pd.DataFrame()

    print("concat sentiments ...")
    resultat = pd.concat(dfs, ignore_index=True)

    print("=== Sauvegarde ===")
    resultat.to_parquet(filename, compression="zstd")

    return resultat


# ============================================================
# 3. Lecture + filtrage des messages (streaming)
# ============================================================

def _load_one_file_messages(args):
    path, message_ids = args

    try:
        m = pd.read_csv(
            path,
            usecols=["message_id", "message_body"],
            engine="python",
            on_bad_lines="skip"
        )

        m["message_id"] = pd.to_numeric(m["message_id"], errors="coerce")
        m = m.dropna(subset=["message_id"])
        m = m[m["message_id"] % 1 == 0] # pour pas garder les ids avec des floats
        m["message_id"] = m["message_id"].astype("Int64")

        result = m[m["message_id"].isin(message_ids)]
        print(f"[message] Lecture : {path} ok")
        return result

    except Exception as e:
        print(f"[ERREUR] {path} : {e}")
        return pd.DataFrame()


def split_paths(paths, k):
    size = len(paths) // k + 1
    return [paths[i:i+size] for i in range(0, len(paths), size)]

def load_filtered_messages_pack(message_ids, paths):
    with Pool() as pool:
        dfs = pool.map(_load_one_file_messages, [(p, message_ids) for p in paths])

    if len(dfs) == 0:
        return pd.DataFrame()

    print("concat messages pack ...")
    return pd.concat(dfs, ignore_index=True)

def stable_hash(ids):
    s = ",".join(str(i) for i in sorted(ids))
    return hashlib.sha1(s.encode()).hexdigest()

def load_filtered_messages(message_ids):
    k = 10
    filename = "dfm" + stable_hash(message_ids) + ".parquet"

    if os.path.exists(filename):
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)

    paths = glob.glob(MESSAGE_PATH + "msg_*.csv")
    path_groups = split_paths(paths, k)

    packs = []

    i = 0
    for group in path_groups:
        i+=1
        print(f"=== Chargement du pack n°{i} de messages ... ===")
        start = time.time()
        df_pack = load_filtered_messages_pack(message_ids, group)
        elapsed = time.time() - start
        print(f"Pack {i} terminé en {elapsed:.2f} sec")
        packs.append(df_pack)

    print("concat final ...")
    start = time.time()
    result = pd.concat(packs, ignore_index=True)
    elapsed = time.time() - start
    print(f"Concat final terminé en {elapsed:.2f} sec")

    print("=== Sauvegarde ===")
    result.to_parquet(filename, compression="zstd")

    return result


# ============================================================
# 4. Pipeline complet
# ============================================================

def generate_dataframe_sentiment(symbols):
    print("=== Chargement des sentiments filtrés ===")
    sentiments = load_filtered_sentiments(symbols)

    assert not sentiments.empty, "Aucun sentiment trouvé pour ces symboles."

    print("=== Chargement des messages filtrés ===")
    message_ids = set(sentiments["message_id"].unique())
    messages = load_filtered_messages(message_ids)

    print("=== Jointure messages + sentiments ===")
    df = messages.merge(sentiments, on="message_id", how="inner")

    # --------------------------------------------------------
    # Normalisation des types
    # --------------------------------------------------------

    print("=== Normalisation des types ===")

    # created_at → en day int16
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["day"] = (df["created_at"] - START_PERIOD_PANDA).dt.days.astype("int16")
    df["day"] = df["day"].fillna(-1).astype("int16")
    df = df.drop(columns=["created_at"])

    # user_id → int32
    df["user_id"] = df["user_id"].astype("int32")

    # sentiment → bool
    df["sentiment"] = df["sentiment"].map({"Bullish": True, "Bearish": False}).astype("bool")

    # --------------------------------------------------------
    # Normalisation de symbol_list → une ligne par symbole 
    # --------------------------------------------------------

    print("=== Explosion des symboles ===")
    df["symbol_list"] = df["symbol_list"].apply(parse_symbols)
    df = df.explode("symbol_list")
    df = df.rename(columns={"symbol_list": "symbol"})
    df = df[df["symbol"].isin(symbols)]

    # --------------------------------------------------------
    # Nettoyage du texte
    # --------------------------------------------------------

    print("=== Nettoyage du texte ===")
    df["text_clean"] = df["message_body"].astype("string").apply(clean_text)

    # --------------------------------------------------------
    # Dataset final compact
    # --------------------------------------------------------

    final_df = df[[
        "user_id",
        "day",
        "symbol",
        "sentiment",
        "text_clean",
    ]]

    return final_df


def generate_dataframe_finance(symbols):
    tickers = yf.Tickers(symbols)

    # Vérification des tickers invalides via fast_info
    tickers_without_desc = [t.ticker for t in tickers.tickers.values() if t.fast_info is None]
    assert len(tickers_without_desc) == 0, f"Ticker(s) inconnu(s) : {tickers_without_desc}"

    # Télécharger les cours
    filename = f"dfc_{'_'.join(symbols)}.parquet"
    if os.path.exists(filename):
            print(f"=== Fichier déjà existant : {filename} ===")
            df = pd.read_parquet(filename)
    else:
        df = tickers.download(start=START_PERIOD, end=END_PERIOD)
        print("=== Sauvegarde ===")
        df.to_parquet(filename, compression="zstd")

    # Calculer l'index "day"
    delta = df.index - START_PERIOD_PANDA
    df["day"] = delta.days.astype("int16")

    # Extraire uniquement les colonnes Close
    df_close = df.xs("Close", level=0, axis=1)

    # IMPORTANT : garder day comme colonne AVANT stack
    df_close = df_close.assign(day=df["day"])

    # stack propre : day reste une colonne, pas un index
    df_final = df_close.melt(id_vars="day", var_name="ticker", value_name="Close")

    # Ajout des prévisions
    df_final["J+1"]  = (df_final.groupby("ticker")["Close"].shift(-1)  > df_final["Close"])
    df_final["J+3"]  = (df_final.groupby("ticker")["Close"].shift(-3)  > df_final["Close"])
    df_final["J+7"]  = (df_final.groupby("ticker")["Close"].shift(-7)  > df_final["Close"])
    df_final["J+30"] = (df_final.groupby("ticker")["Close"].shift(-30) > df_final["Close"])

    return df_final



def generate_dataframe(symbols):
    filename = f"df_{'_'.join(symbols)}.parquet"

    if os.path.exists(filename):
        print(f"=== Fichier déjà existant : {filename} ===")
        return pd.read_parquet(filename)

    print("=== Génération ===")
    df_sentiment = generate_dataframe_sentiment(symbols)
    print("df de generate_dataframe_sentiment OK")
    #print(df_sentiment["day"])
    df_finance = generate_dataframe_finance(symbols)
    print("df de generate_dataframe_finance OK")
    #print(df_finance["day"])    
    resultat = df_sentiment.merge(
        df_finance,
        left_on=["day", "symbol"],
        right_on=["day", "ticker"],
        how="inner"
    )
    resultat = resultat.drop(columns=["symbol"])

    print("=== Sauvegarde ===")
    resultat.to_parquet(filename, compression="zstd")

    print(f"=== Sauvegardé dans {filename} ===")
    return resultat


# ============================================================
# Pour tester (attention c'est long !)
# ============================================================

if __name__ == "__main__":
    SYMBOLS = ["TSLA","AAPL"]
    df = generate_dataframe(SYMBOLS)
    print(df.head())
