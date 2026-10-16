import jax
import jax.numpy as jnp
import time
import optax

class Trainer:
    def __init__(self, model, rng, input_shape, learning_rate=0.001):
        self.model = model
        self.rng = rng = jax.random.PRNGKey(int(time.time()))
        model.init_state(rng, input_shape, learning_rate)

    @staticmethod
    @jax.jit
    def train_step(state, x, y, rng):
        rng, dropout_rng = jax.random.split(rng)
        weights = jnp.array([3.0, 1.0])   # punition ×3 pour la classe rare
        def loss_fn_local(params):
            probs = state.apply_fn(params, x, train=True, rngs={"dropout": dropout_rng})
            #y_onehot = jax.nn.one_hot(y, 2)

            ce = -jnp.log(probs[jnp.arange(len(y)), y] + 1e-9)
            weighted_ce = ce * weights[y]
            return jnp.mean(weighted_ce)

        loss, grads = jax.value_and_grad(loss_fn_local)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, rng, loss

    def train_epoch(self, x_data, y_data, rng, batch_size=256):
        state = self.model.get_state()

        n = len(x_data)
        num_batches = n // batch_size

        losses = []

        for i in range(num_batches):
            x_batch = x_data[i*batch_size:(i+1)*batch_size]
            y_batch = y_data[i*batch_size:(i+1)*batch_size]

            state, rng, loss = Trainer.train_step(state, x_batch, y_batch, rng)
            losses.append(float(loss))

        self.model.state = state
        return losses
    
    def fit(self, X, Y, batch_size=256, epochs=5):
        rng = self.rng
        ko = 0
        LR = self.model.initial_learning_rate
        loss = 100
        fin = False
        for epoch in range(epochs):
            losses = self.train_epoch(X, Y, rng, batch_size)
            previous_loss = loss
            loss = sum(losses) / len(losses)
            gain = previous_loss - loss
            if gain<0:
                ko+=1
            else:
                ko=0
            print(f"Epoque {epoch} {loss}")
            if ko>=3:
                if fin:
                    return
                else:
                    print("new LR !")
                    LR = LR / 2
                    new_tx = optax.adam(LR)
                    self.model.state = self.model.state.replace(tx=new_tx)
                    ko=0
                    fin = LR < self.model.initial_learning_rate / 5

if __name__ == "__main__":
    print(jax.devices())    