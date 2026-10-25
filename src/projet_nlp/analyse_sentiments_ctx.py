from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import train_embedding_model, add_generated_sentiment, add_user_embedding, ROBERTA_S, add_message_sentiment
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model
import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Chargement des sentiments

tickers = [
    "TSLA",
    "AAPL",
    "AMD",
    "NVDA",
    "GOOGL",
    "META"
]

#tickers = ["TSLA","AAPL"]

print("get DF for sentiment analysis")
df = get_df_for_sentiment_analysis(tickers)

df.head(5)["message_id"]

print("add user embedding")
df = add_user_embedding(df, regenerer=False ,model_name=ROBERTA_S)

print(df.shape)
print(df.head(7)[["sentiment", "message", "message_id"]])


print("add_message_sentiment withCtx=True")
resultat, model = add_message_sentiment(df, regenerer=False , model_name=ROBERTA_S, taille_max = 1499999, withCtx=2, test=True)
#print(resultat.head(5))
#print("")
print("fin")