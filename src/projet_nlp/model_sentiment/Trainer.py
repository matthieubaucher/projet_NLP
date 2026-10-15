import jax
import jax.numpy as jnp

class Trainer:
    def __init__(self, model, rng, input_shape, learning_rate=0.001):
        self.model = model
        model.init_state(rng, input_shape, learning_rate)

    @staticmethod
    @jax.jit
    def train_step(state, x, y):
        weights = jnp.array([2.0, 1.0])   # punition ×4 pour la classe rare
        def loss_fn_local(params):
            probs = state.apply_fn(params, x)
            y_onehot = jax.nn.one_hot(y, 2)

            ce = -jnp.log(probs[jnp.arange(len(y)), y] + 1e-9)
            weighted_ce = ce * weights[y]
            return jnp.mean(weighted_ce)

        loss, grads = jax.value_and_grad(loss_fn_local)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train_epoch(self, x_data, y_data, batch_size=256):
        state = self.model.get_state()

        n = len(x_data)
        num_batches = n // batch_size

        losses = []

        for i in range(num_batches):
            x_batch = x_data[i*batch_size:(i+1)*batch_size]
            y_batch = y_data[i*batch_size:(i+1)*batch_size]

            state, loss = Trainer.train_step(state, x_batch, y_batch)
            losses.append(float(loss))

        self.model.state = state
        return losses
    
    def fit(self, X, Y, batch_size=256, epochs=5):
        for epoch in range(epochs):
            print(f"Epoque {epoch}")
            self.train_epoch(X, Y, batch_size)

if __name__ == "__main__":
    print(jax.devices())    