# Bare minimum implementation of absolute sinusoidal position embedding. NOT PRODUCTION READY. WILL IMPROVE IT STEP BY STEP.

import numpy as np

np.random.seed(42)

# Sinusoidal Position Embedding Formula

# p(k,i) = sin(k/10000^(2i/d)) if i is even, else cost(k/10000^(2i/d))

# k is the position of the token (2nd token, 3rd token etc. in a sequence of sentences)
# i is the dimension index (1st dimension, 2nd dimension etc. in a vector of embeddings)
# d is the dimension of the embedding


total_tokens = 3 # 3 words
d = 5 # 5 dimensions . ex. embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

base_embedding = np.random.randn(total_tokens, d)

print(f"Base embedding: shape = {base_embedding.shape}\n{base_embedding}")

assert base_embedding.shape == (total_tokens, d)

def pos_embedding(k, i, d):
    if i%2 == 0:
        pos_embedding_offset = np.sin(k/10000**(i/d))
    else:
        pos_embedding_offset = np.cos(k/10000**((i-1)/d)) # The logic is that for a given position k, the even dimensions are sin and the odd dimensions are cos of the same frequency..

    return pos_embedding_offset

sample_pos_embedding_offset = pos_embedding(k=5, i=5, d=d)
print(f"Pos embedding offset: {sample_pos_embedding_offset}")


def abs_pos_embedding(d, total_tokens):
    pos_embedding_matrix = np.zeros((total_tokens, d))
    for k in range(total_tokens):
        for i in range(d):
            pos_embedding_matrix[k][i] = pos_embedding(k=k, i=i, d=d)

    return pos_embedding_matrix

abs_pos_embedding = abs_pos_embedding(d=d, total_tokens=total_tokens)

embedding_with_pos = base_embedding + abs_pos_embedding
print(f"Position embedding: shape = {abs_pos_embedding.shape}\n{abs_pos_embedding}")

print(f"Embedding with pos: shape = {embedding_with_pos.shape}\n{embedding_with_pos}")