from flax import linen as nn
from projet_nlp.model_sentiment.Modele import Modele

class MLP(Modele):
    hidden_sizes: tuple = (128, 64, 64)

    @nn.compact
    def __call__(self, x, train=True):
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.Dropout(0.20)(x, deterministic=not train)
            x = nn.relu(x)
        logits = nn.Dense(self.n_outputs)(x)
        return nn.softmax(logits)

