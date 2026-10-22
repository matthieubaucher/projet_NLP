from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import train_embedding_model, add_generated_sentiment, MPNET, FINBERT_TONE, ROBERTA_S,GOOGLE, MINI_BERTA, ALBERT, GROS_GOOGLE, ROBERTA, add_message_sentiment
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model
import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")

#MPNET :        10_000 -> 71.8%  100_000 -> 70.5%   400_000 -> 83.0%  1_000_000 -> coll.  2_000_000 -> 81.7%
#MINI_BERTA :   10_000 -> 69.1%  100_000 -> 77.7%   400_000 -> 77.9% 
#ALBERT :       10_000 -> 70.5%  100_000 -> 72.6%                     
#GROS_GOOGLE :  10_000 -> 71.8%  100_000 -> 78.9%   400_000 -> 79.8%  1_000_000 -> 81.3%  2_000_000 -> 82.4%  
#GOOGLE :       10_000 -> 61.1%  100_000 -> 72.9%   400_000 -> 72.8%
#ROBERTA_S :    10_000 -> 70.5%  100_000 -> 80.7%   400_000 -> 82.2%  1_000_000 -> 82.3%  2_000_000 -> 82.9%   <-winner
#ROBERTA :      10_000 -> 71.8%  100_000 -> 78.5%   400_000 -> 80.8%  1_000_000 -> 79.6%  2_000_000 -> 82.9%  
#FINBERT-TONE : 10_000 -> 66.4%  100_000 -> 75.4%   400_000 -> 77.0%  1_000_000 -> 79.3%  2_000_000 -> 82.0%
resultat, model = add_message_sentiment(df, regenerer=False , model_name=ROBERTA_S, taille_max = 2000000)
print(resultat.head())
print("")


rows = resultat[(resultat["sentiment"]!=resultat["generated_sentiment"]) & (resultat["sentiment"]!="UNKNOWN")].iloc[0:30]
print(rows)