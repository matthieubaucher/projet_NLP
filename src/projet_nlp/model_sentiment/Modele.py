import jax
from flax import linen as nn
from flax.training import train_state
import optax
import jax.numpy as jnp
import numpy as np

class Modele(nn.Module):
    n_outputs: int = 2
    state: train_state.TrainState = None   # poids + état de l'optimiseur

    def init_state(self, rng, input_shape, learning_rate):
        rng, init_rng = jax.random.split(rng)
        init_x = jax.random.normal(init_rng, input_shape)

        # init crée les paramètres du réseau
        params = self.init(init_rng, init_x)

        # optimiseur
        tx = optax.adam(learning_rate)

        # TrainState = params + apply_fn + optimiseur
        self.state = train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=tx,
        )
        return self.state

    def get_state(self):
        assert self.state is not None, "vous devez initialiser l'état pour l'entraînement !"
        return self.state

    def predict(self, X):
        apply_fn = self.state.apply_fn
        params = self.state.params

        # Probabilités directement
        probs = jax.vmap(lambda x: apply_fn(params, x))(X)  # (N, 2)

        # Classe = argmax
        classes = jnp.argmax(probs, axis=1)

        classes = np.array(classes).astype(int)
        return classes