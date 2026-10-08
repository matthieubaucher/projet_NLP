from projet_nlp.df_generator import generate_dataframe
from projet_nlp.df_cleaner import clean_df
# Pour que ca marche :
# 1) Le répertoire 'data' doit contenir les données sur les sentiments et les messages dans data/sentiments et data/messages
# 2) Se mettre dans le dossier projet_nlp avant de lancer get_df_for_sentiment_analysis ou get_df_for_forecast_analysis
# Le dataframe est sauvegardé avec un nom de fichier basé sur les symboles, donc la première exécution pour ces symboles la
# est lente (21 minutes environ), et la seconde fois est très rapide (quelques secondes).



# Retourne un dataframe pour pouvoir faire une analyse de sentiment dessus
#
# IN  : une liste de symbole de ticker (ex ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA"])
# OUT : un dataframe directement prêt à être utilisé avec juste ce qu'il faut comme colonnes
#
def get_df_for_sentiment_analysis(symbols):
    df = clean_df(generate_dataframe(symbols))
    df.drop(columns=["J+1","J+3","J+7","J+30"], inplace=True)
    return df

# Retourne un dataframe pour pouvoir faire une analyse de prévision de cours à partir des sentiment dessus
# Certains sentiments ne sont pas présents car non renseignés par l'utilisateur
#
# IN  : une liste de symbole de ticker (ex ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA"])
# OUT : un dataframe directement prêt à être utilisé avec juste ce qu'il faut comme colonnes
#
def get_df_for_forecast_analysis(symbols):
    df = clean_df(generate_dataframe(symbols))
    df.drop(columns=["message"], inplace=True)
    return df


# ============================================================
# Pour tester : 
# ============================================================

if __name__ == "__main__":

    df = clean_df(generate_dataframe(["TSLA","AAPL"]))

    print(df.describe())
    print(df.info())
    print(df.head(5))

    #select = df[(df["user_id"] == 300155) & (df["ticker"] == 'TSLA')]
    #select = df[ (df["message"].str.contains("\$tsla", case=False, na=False)) & (df["message"].str.contains("\$aapl", case=False, na=False))]
    #print(select.head(5))
