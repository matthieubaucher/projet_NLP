from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis 
from projet_nlp.df_decorator import train_embedding_model, add_generated_sentiment, FINBERT_TONE, ROBERTA
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model

# Chargement des sentiments

tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)
print(df["sentiment"].value_counts(dropna=False))
print(df.head())
print("")


# Création des dataset de train et de test

#1_000 (194) -> 63%  10_000 (1944) -> 62%  100_000 (19000) -> 72%   400_000 (73192)-> 77%  1_000_000 -> ?

N = 1000000
model_name = FINBERT_TONE  # TODO : tester avec plus de tweet et tester Roberta qui est plus orienté tweet et plus simple
fn = "finebert-tone"

df = df.sample(n=N, random_state=42)
df_train, df_test = train_test_split(
    df,
    test_size=0.2,      # 20% pour le test
    random_state=42,    
    shuffle=True
)
df_test, df_validation = train_test_split(
    df_test,
    test_size=0.2,      # 20% pour la validation inter époque
    random_state=42,    
    shuffle=True
)


# Entrainement du modele 

print("TRAIN")
nom_sauvegarde = f"toto_{fn}{N}"
model = train_embedding_model(df_train, nom_sauvegarde=nom_sauvegarde, df_validation=df_validation, model_name=model_name)


# Utilisation du modele 

print("TEST")
df_test, score = add_generated_sentiment(df_test, model)

cols = ["ticker", "sentiment", "generated_sentiment"]
print(df_test[cols].head(10).to_string())
print(score)



# petit essai pour voir si la sauvegarde fonctionne :
model.save()
model = load_model(nom_sauvegarde)
df_test, score = add_generated_sentiment(df_train, model)
print(df_test[cols].head(10).to_string())
print(score) # on peut voir si le résultat est meilleur sur la donnée d'entrainement (surapprentissage)