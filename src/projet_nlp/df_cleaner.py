
import pandas as pd
def clean_df(df):
    df.rename(columns={"text_clean": "message"}, inplace=True)
    df.drop(columns=["Close"], inplace=True)
    df["user_id"] = df["user_id"].astype(pd.CategoricalDtype(ordered=False))
    df["day"] = df["day"].astype(pd.CategoricalDtype(ordered=True))

    return df



