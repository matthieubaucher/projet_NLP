from projet_nlp.df_generator import generate_dataframe

df = generate_dataframe(["TSLA","AAPL"])

print(df.describe())
print(df.head(5))