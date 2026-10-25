import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from projet_nlp.model_sentiment.EmbeddingClassifier import EmbeddingClassifier
from projet_nlp.model_sentiment.EmbeddingClassifierCtx import EmbeddingClassifierCtx
from sklearn.model_selection import train_test_split
from projet_nlp.model_sentiment.EmbeddingClassifier import load_model
import html
from projet_nlp.df_cleaner import filter_bad_phrases

model = SentenceTransformer("cardiffnlp/twitter-roberta-base-sentiment", device="cuda")

bad_text = "$toto $toto sooner or later you will see the sells in goog and money redistributed to aapl, they will bring down goog & up aapl "

batch = [
    "hello world",
    bad_text,
    "another safe phrase"
]

emb = model.encode([bad_text]*64, convert_to_tensor=True, device="cuda")



print(emb)
