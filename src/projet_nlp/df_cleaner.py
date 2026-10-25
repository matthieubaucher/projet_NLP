import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def clean_df(df):
    df.rename(columns={"text_clean": "message"}, inplace=True)
    df.drop(columns=["Close"], inplace=True)
    df["user_id"] = df["user_id"].astype(pd.CategoricalDtype(ordered=False))
    df["day"] = df["day"].astype(pd.CategoricalDtype(ordered=True))

    # Anonymiser les tag pour ne pas entrainer dessus
    df["message"] = df["message"].str.replace(r"\$[A-Za-z0-9_]+", "$toto", regex=True)

    return df

def filter_bad_phrases(df, model_name, text_col="message"):
    tok = AutoTokenizer.from_pretrained(model_name)

    good_rows = []
    bad_rows = []

    for idx in tqdm(range(len(df)), desc="Tokenisation CPU"):
        text = df.iloc[idx][text_col]
        try:
            tok.encode(text)
            good_rows.append(idx)
        except Exception as e:
            print(f"KO phrase index={idx} → {repr(text)}")
            print("Reason:", e)
            bad_rows.append(idx)

    df_good = df.loc[good_rows].reset_index(drop=True)
    df_bad  = df.loc[bad_rows].reset_index(drop=True)

    return df_good, df_bad



