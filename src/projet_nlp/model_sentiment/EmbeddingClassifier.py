import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
import time
import os

# Réseau de neurone entrainable. prend un modèle Sentence->Embedding et rajoute une tête de classification
class EmbeddingClassifier(nn.Module):
    def __init__(self, model_name="yiyanghkust/finbert-tone", nom_sauvegarde = "finbert-tone"):
        super().__init__()

        self.nom_sauvegarde = nom_sauvegarde

        # On récupère le transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        # On créé la tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 2) #256),
            #nn.Tanh(),
            #nn.Dropout(0.2),
            #nn.Linear(256, 2),
            #nn.Softmax(dim=-1)
        )

    def forward(self, tokens):
        out = self.encoder(**tokens)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

    def tokenize(self, trainset):
        if trainset is None:
            return None

        sentences = []
        labels = []

        for sentence, label in trainset:
            sentences.append(sentence)
            labels.append(label)

        # Tokenisation en une seule fois
        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        ).to("cuda")

        # Reconstruction du dataset tokenisé
        trainset = [
            (
                {
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i]
                },
                labels[i]
            )
            for i in range(len(labels))
        ]

        return trainset


    def fit(self, trainset, finetune= True, epochs=5, batch_size=64, lr=1e-5, trainset_validation=None):

        # refaire le trainset avec des tokens cette fois
        trainset = self.tokenize(trainset)
        trainset_validation = self.tokenize(trainset_validation)

        dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(reduction="none") # loss par élément
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for param in self.encoder.parameters():
            param.requires_grad = finetune

        for epoch in range(epochs):
            self.train()
            elapse = time.time()
            total_loss = 0
            for sentences, labels in dataloader:
                labels = labels.to("cuda")
                with torch.amp.autocast("cuda"):
                    outputs = self(sentences)
                    losses = criterion(outputs, labels)  
                    #losses = torch.clamp(losses, max=20.0)                  # clamp individuel
                    loss = losses.mean()                                   # moyenne finale
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            elapse = time.time() - elapse
            score = 0
            if trainset_validation is not None:
                score = self.test(trainset_validation)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - score: {score:.2f} - Elapse: {elapse:.1f}")

    def entraine(self, trainset, epochs, epochfinetune, lr=1e-5, trainset_validation=None):
        if epochs>0:
            self.fit(trainset, finetune=False, epochs=epochs, lr=lr, trainset_validation=None)
        self.fit(trainset, finetune=True, epochs=epochfinetune, lr=lr, trainset_validation=trainset_validation)       

    def test(self, validationset_tokenized):
        self.eval()
        dataloader = DataLoader(validationset_tokenized, batch_size=128, shuffle=True)
        with torch.no_grad():
            ok = 0
            for sentences, labels in dataloader:
                labels_vrai = labels.to("cpu")
                probas = self(sentences)
                labels_predit = torch.argmax(probas, dim=1).cpu().numpy()
                ok += (labels_predit == labels_vrai).sum()
            return 100 * ok / len(validationset_tokenized)

    def predict(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda") 
        self.eval()
        with torch.no_grad():
            probas = self(tokens)
            labels = torch.argmax(probas, dim=1).cpu().numpy()
            mapping = np.array(["BAISSE", "HAUSSE"])
            sentiments = mapping[labels]
            return sentiments

    def save(self):
        torch.save(self.state_dict(), f"{self.nom_sauvegarde}.pt")

def load_model(nom_sauvegarde = "finbert-tone"):
    filename = f"{nom_sauvegarde}.pt"
    if not os.path.exists(filename):
        return None
    model = EmbeddingClassifier()
    model.load_state_dict(torch.load(filename))
    model.to("cuda")
    model.eval()   
    return model     