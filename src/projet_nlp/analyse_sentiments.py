from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import add_message_embedding

# chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())

# construction du model pour transformer les tweets en embeddings (1h15 de calculs estimé)

add_message_embedding(df)

# pour voir ce que cela donne ...
print(df.head())

# TODO : transformer les embeddings en label et comparer
