import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import copy
import os

class EmbeddingClassifierCtx2(nn.Module):
    def __init__(self, model_name, nom_sauvegarde):
        super().__init__()

        self.nom_sauvegarde = nom_sauvegarde

        # Transformer principal
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Deux copies de la dernière couche pour user et user_previous
        self.user_encoder = copy.deepcopy(self.encoder.encoder.layer[-1])
        self.user_prev_encoder = copy.deepcopy(self.encoder.encoder.layer[-1])

        # Dimensions réduites
        N = 2
        K = 2

        self.layer_user = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, N),
            nn.Tanh()
        )

        self.layer_user_prev = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, K),
            nn.Tanh()
        )

        # Tête finale : msg + user + user_previous
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size + N + K, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, tokens, user_embeddings, user_prev_embeddings):
        # Embedding message
        out = self.encoder(**tokens)
        resultat_msg = out.last_hidden_state[:, 0, :]

        # Embedding user global
        user_embeddings = user_embeddings.unsqueeze(1)
        att = torch.ones(user_embeddings.size(0), 1).to(user_embeddings.device)

        user_hidden = self.user_encoder(
            user_embeddings,
            attention_mask=att
        )[0][:, 0, :]

        resultat_user = self.layer_user(user_hidden)

        # Embedding user_previous
        user_prev_embeddings = user_prev_embeddings.unsqueeze(1)
        att_prev = torch.ones(user_prev_embeddings.size(0), 1).to(user_prev_embeddings.device)

        user_prev_hidden = self.user_prev_encoder(
            user_prev_embeddings,
            attention_mask=att_prev
        )[0][:, 0, :]

        resultat_user_prev = self.layer_user_prev(user_prev_hidden)

        # Fusion
        final = torch.cat([resultat_msg, resultat_user, resultat_user_prev], dim=-1)

        return self.classifier(final)

    def tokenize(self, trainset, token_size_max=256):
        if trainset is None:
            return None

        sentences = []
        labels = []
        user_embeddings = []
        user_prev_embeddings = []

        for sentence, user_emb, user_prev_emb, label in trainset:
            sentences.append(sentence)
            labels.append(label)
            user_embeddings.append(user_emb)
            user_prev_embeddings.append(user_prev_emb)

        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=token_size_max
        ).to("cuda")

        labels = torch.tensor(labels, dtype=torch.long).to("cuda")
        user_embeddings = torch.from_numpy(np.array(user_embeddings, dtype=np.float32)).to("cuda")
        user_prev_embeddings = torch.from_numpy(np.array(user_prev_embeddings, dtype=np.float32)).to("cuda")

        trainset = [
            (
                {
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i]
                },
                user_embeddings[i],
                user_prev_embeddings[i],
                labels[i]
            )
            for i in range(len(labels))
        ]

        return trainset

    def fit(self, trainset, finetune=True, epochs=5, batch_size=64, lr=1e-5, trainset_validation=None, token_size_max=256):
        trainset = self.tokenize(trainset, token_size_max)
        trainset_validation = self.tokenize(trainset_validation, token_size_max)

        dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-7)

        for param in self.encoder.parameters():
            param.requires_grad = finetune

        for epoch in range(epochs):
            self.train()
            elapse = time.time()
            total_loss = 0

            for sentences, user_embeddings, user_prev_embeddings, labels in dataloader:
                labels = labels.to("cuda")

                with torch.amp.autocast("cuda"):
                    outputs = self(sentences, user_embeddings, user_prev_embeddings)
                    losses = criterion(outputs, labels)
                    loss = losses.mean()

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
        if epochs > 0:
            self.fit(trainset, finetune=False, epochs=epochs, lr=lr, trainset_validation=None)
        self.fit(trainset, finetune=True, epochs=epochfinetune, lr=lr, trainset_validation=trainset_validation)

    def test(self, validationset_tokenized):
        self.eval()
        dataloader = DataLoader(validationset_tokenized, batch_size=128, shuffle=True)

        with torch.no_grad():
            ok = 0
            for sentences, user_embeddings, user_prev_embeddings, labels in dataloader:
                labels_vrai = labels.to("cpu")
                probas = self(sentences, user_embeddings, user_prev_embeddings)
                labels_predit = torch.argmax(probas, dim=1).cpu().numpy()
                ok += (labels_predit == labels_vrai).sum()

            return 100 * ok / len(validationset_tokenized)

    def predict(self, sentences, user_embeddings, user_prev_embeddings):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=256).to("cuda")

        if isinstance(user_embeddings, list):
            user_embeddings = torch.from_numpy(np.array(user_embeddings, dtype=np.float32)).to("cuda")

        if isinstance(user_prev_embeddings, list):
            user_prev_embeddings = torch.from_numpy(np.array(user_prev_embeddings, dtype=np.float32)).to("cuda")

        self.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            probas = self(tokens, user_embeddings, user_prev_embeddings)
            labels = torch.argmax(probas, dim=1).cpu().numpy()
            mapping = np.array(["BAISSE", "HAUSSE"])
            return mapping[labels]

    def save(self):
        torch.save(self.state_dict(), f"{self.nom_sauvegarde}")

def load_model(nom_sauvegarde, model_name):
    filename = f"{nom_sauvegarde}"
    if not os.path.exists(filename):
        return None
    model = EmbeddingClassifierCtx(model_name=model_name, nom_sauvegarde=nom_sauvegarde)
    model.load_state_dict(torch.load(filename))
    model.to("cuda")
    model.eval()
    return model
