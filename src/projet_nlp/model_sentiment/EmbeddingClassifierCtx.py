import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
import time
import os
import copy

# Classification avec msg + user
class EmbeddingClassifierCtx(nn.Module):
    def __init__(self, model_name, nom_sauvegarde):
        super().__init__()

        self.nom_sauvegarde = nom_sauvegarde

        # On récupère le transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.user_encoder = copy.deepcopy(self.encoder.roberta.encoder.layer[-1]) # supprimer "roberta" pour finebertt-one
        # On créé les têtes de classification
        N = 2
        self.layer_user = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, N),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
                    nn.Linear(self.encoder.config.hidden_size+N, 128), 
                    nn.Tanh(),
                    #nn.Dropout(0.2),
                    nn.Linear(128, 2)
                )
        

    def forward(self, tokens, user_embeddings):
        # Sortie du transformer msg
        out = self.encoder(**tokens)
        resultat_msg = out.last_hidden_state[:, 0, :]

        # Entrées du transformer user
        user_embeddings = user_embeddings.unsqueeze(1)  # (B, 1, H)
        att = torch.ones(user_embeddings.size(0), 1).to(user_embeddings.device)
        # Sortie du transformer user
        user_hidden = self.user_encoder(
            user_embeddings,
            attention_mask=att
        )[:, 0, :] 
        # Réduction vers un espace latent qui généralise car plus petit
        resultat_user = self.layer_user(user_hidden)

        # Fusion des 2 sorties
        final = torch.cat([resultat_msg, resultat_user], dim=-1)

        # Classification avec la tête 
        return self.classifier(final)

    def tokenize(self, trainset):
        if trainset is None:
            return None

        sentences = []
        labels = []
        user_embeddings = []

        for sentence, label, user_embedding in trainset:
            sentences.append(sentence)
            labels.append(label)
            user_embeddings.append(user_embedding)

        # Tokenisation en une seule fois
        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to("cuda")

        # Reconstruction du dataset tokenisé
        trainset = [
            (
                {
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i]
                },
                user_embeddings[i],
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
            for sentences, user_embeddings, labels in dataloader:
                labels = labels.to("cuda")
                with torch.amp.autocast("cuda"):
                    outputs = self(sentences, user_embeddings)
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
            for sentences, user_embeddings, labels in dataloader:
                labels_vrai = labels.to("cpu")
                probas = self(sentences, user_embeddings)
                labels_predit = torch.argmax(probas, dim=1).cpu().numpy()
                ok += (labels_predit == labels_vrai).sum()
            return 100 * ok / len(validationset_tokenized)

    def predict(self, sentences, user_embeddings):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda") 
        self.eval()
        with torch.no_grad():
            probas = self(tokens, user_embeddings)
            labels = torch.argmax(probas, dim=1).cpu().numpy()
            mapping = np.array(["BAISSE", "HAUSSE"])
            sentiments = mapping[labels]
            # Nettoyage GPU
            del tokens
            del probas
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return sentiments

    def save(self):
        torch.save(self.state_dict(), f"{self.nom_sauvegarde}")

def load_model(nom_sauvegarde, model_name):
    filename = f"{nom_sauvegarde}"
    if not os.path.exists(filename):
        return None
    model = EmbeddingClassifierCtx(nom_sauvegarde=nom_sauvegarde, model_name=model_name)
    model.load_state_dict(torch.load(filename))
    model.to("cuda")
    model.eval()   
    return model     