from projet_nlp.df_generator import generate_dataframe
from projet_nlp.df_cleaner import clean_df
# Pour que ca marche :
# 1) Le répertoire 'data' doit contenir les données sur les sentiments et les messages dans data/sentiments et data/messages
# 2) Se mettre dans le dossier projet_nlp avant de lancer get_df_for_sentiment_analysis ou get_df_for_forecast_analysis
# Le dataframe est sauvegardé avec un nom de fichier basé sur les symboles, donc la première exécution pour ces symboles la
# est lente (45 minutes environ sur AAPL+TSLA), et la seconde fois est très rapide (quelques secondes).


# Retourne un dataframe pour pouvoir faire une analyse de sentiment dessus
# Certains sentiments ne sont pas présents car non renseignés par l'utilisateur
#
# IN  : une liste de symbole de ticker (ex ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA"])
# OUT : un dataframe directement prêt à être utilisé avec juste ce qu'il faut comme colonnes
#
def get_df_for_sentiment_analysis(symbols):
    df = clean_df(generate_dataframe(symbols, keep_wend_msg=True))
    df.drop(columns=["J+1","J+3","J+7","J+30"], inplace=True)
    return df

# Retourne un dataframe pour pouvoir faire une analyse de prévision de cours à partir des sentiment dessus
# Certains sentiments ne sont pas présents car non renseignés par l'utilisateur
# Les week-ends on change la date des messages pour les mettre le vendredi : si vous n'êtes pas d'accord 
# avec cela (car ca peut être considéré comme une approximation), il faut mettre keep_wend_msg à False
# et dans ce cas, on les ignore
#
# IN  : - une liste de symbole de ticker (ex ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA"])
#       - booléen pour dire si on garde ou pas les messages du week-end (oui par défaut)   
# OUT : un dataframe directement prêt à être utilisé avec juste ce qu'il faut comme colonnes
#
def get_df_for_forecast_analysis(symbols, keep_wend_msg=True):
    df = clean_df(generate_dataframe(symbols, keep_wend_msg))
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
