from projet_nlp.df_generator import generate_dataframe

# Pour que ca marche :
# 1) Le répertoire data doit contenir les données sur les sentiments et les messages dans data/sentiments et data/messages
# 2) Se mettre dans le dossier projet_nlp/src/projet_nlp avabt de lancer ce script
df = generate_dataframe(["TSLA","AAPL"]) # on spécifie les symboles que l'on veut dans le résultat final
# Le dataframe est sauvegardé avec un nom de fichier basé sur les symboles, donc la première exécution pour ces symboles la
# est lente (35 minutes ?), et la seconde fois est très rapide.

print(df.describe())
print(df.head(5))

