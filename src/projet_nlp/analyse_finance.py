from projet_nlp.dataframe_generator import get_df_for_forecast_analysis 
from projet_nlp.df_decorator import ROBERTA_S, add_message_sentiment
import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Chargement des données générales pour de la finance
tickers = ["TSLA","AAPL"]
df = get_df_for_forecast_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")

# Ajout des sentiments générés
resultat, model = add_message_sentiment(df, regenerer=True , model_name=ROBERTA_S, taille_max = 2000000, test=False)
resultat.drop(columns=["message"], inplace=True) # Pas besoin de cette colonne pour la finance
petit = resultat.head(5)
print(resultat.head())
print("")

