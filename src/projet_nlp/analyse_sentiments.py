from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import train_embedding_model, add_generated_sentiment
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import EmbeddingClassifier, load_model

# Chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")


# Création des dataset de train et de test

df = df.sample(n=10000, random_state=42)
df_train, df_test = train_test_split(
    df,
    test_size=0.2,      # 20% pour le test
    random_state=42,    
    shuffle=True
)


# Entrainement du modele 

print("TRAIN")
model = train_embedding_model(df_train, nom_sauvegarde="toto")


# Utilisation du modele 

print("TEST")
df_test, err = add_generated_sentiment(df_test, model)

cols = ["ticker", "sentiment", "generated_sentiment"]
print(df_test[cols].head(10).to_string())
print(err)


# petit essai pour voir si la sauvegarde fonctionne :
#model.save()
#model = load_model("toto")
#df_test, err = add_generated_sentiment(df_train, model)
#print(df_test[cols].head(10).to_string())
#print(err)