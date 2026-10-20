from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import train_embedding_model, add_generated_sentiment, FINBERT_TONE, ROBERTA, add_message_sentiment
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model

# Chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")



#ROBERTA :      1_000 -> 53%  10_000 -> 70%  100_000 -> 76%   400_000 -> 80%  1_000_000 -> ?
#FINBERT-TONE : 1_000 -> 63%  10_000 -> 62%  100_000 -> 72%   400_000 -> 77%  1_000_000 -> ?
df, model = add_message_sentiment(df, regenerer=False , model_name=ROBERTA, taille_max = 1000000)
print(df.head())
print("")

