from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import add_message_embedding, add_generated_sentiment, inference_generated_sentiment
from projet_nlp.model_sentiment.MLP import MLP

# chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")

# construction du model pour transformer les tweets en embeddings (1h15 de calculs estimé)

df = add_message_embedding(df, taille_max = 10000, regenerer=True)

# pour voir ce que cela donne ...

print(df.head(10))
print("")
print("")
print("")

print("TRAIN")
reseau_de_neurones = MLP(hidden_sizes=(128,64))

add_generated_sentiment(df, reseau_de_neurones)

cols = ["ticker", "sentiment", "generated_sentiment"]
print(df[cols].head(10).to_string())
print("")
print("")
print("")

print("TEST")
df = add_message_embedding(df, taille_max = 10000, regenerer=True)
df, err = inference_generated_sentiment(df, reseau_de_neurones)

print(df[cols].head(10).to_string())

print(err)