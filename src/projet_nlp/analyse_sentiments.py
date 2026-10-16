from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import add_message_embedding, train_and_add_generated_sentiment, inference_and_add_generated_sentiment
from projet_nlp.model_sentiment.MLP import MLP

# chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")

# construction du model pour transformer les tweets en embeddings (1h15 de calculs estimé)

df_train = add_message_embedding(df, taille_max = 100000, regenerer=False)

# pour voir ce que cela donne ...

print(df_train.head(10))
print("")
print("")
print("")

print("TRAIN")
reseau_de_neurones = MLP(hidden_sizes=(64,64))

train_and_add_generated_sentiment(df_train, reseau_de_neurones)

cols = ["ticker", "sentiment", "generated_sentiment"]
print(df_train[cols].head(10).to_string())
print("")
print("")
print("")

print("TEST")
df_test = add_message_embedding(df, excluded_df=df_train, taille_max = 10000, regenerer=False)
df_test, err = inference_and_add_generated_sentiment(df_test, reseau_de_neurones)

print(df_test[cols].head(10).to_string())

print(err)