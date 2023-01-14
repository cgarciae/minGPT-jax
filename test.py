from mingpt.model import GPT
import jax.numpy as jnp
import jax

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
model = GPT(model_config)

idx = jnp.ones((2, 10), dtype=jnp.int32)
key = jax.random.PRNGKey(0)
print(model.tabulate(key, idx, train=False))