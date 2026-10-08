from projet_nlp.dataframe_generator import get_df_for_sentiment_analysis    

# quelques essais pour le moment ...
tickers = ["TSLA","AAPL"]
df = get_df_for_sentiment_analysis(tickers)

print(df["sentiment"].value_counts(dropna=False))
