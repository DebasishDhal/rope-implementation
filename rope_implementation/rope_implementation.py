import numpy as np

def apply_rope(x, base=10000):
    """
    x.shape = (seq_len, dim)

    In simple words, do this
    x = [x0,
         x1,
         x2,
         .
         .
         x_seq_len]

    x1 = [x10, x11, x12, x13,...., x_1dim]

    Take pairwise embedding values, rotate them by a certain angle, stich them back together

    theta[k, i] = k * omega[i] # this is the rotation angle
    omega[i] = 1 / 10000^(2i/d) # frequency for dimension pair r
    theta[k, i] = k / (10000 ** (2 * i / d))

    
    """

    seq_len, dim = x.shape
    assert dim%2 == 0

    positions = np.arange(seq_len) # [0,1,2,..,seq_len]

    positions = positions[:, None] #simple broadcasting into columns
    #[[0], [1], [2]...[seq_len]]

    pair_indices = np.arange(start=0, stop=dim, step=2) # [0, 2, ..., dim]

    inv_freq = 1 / (
                base ** (pair_indices / dim)
                )

    angles = positions * inv_freq # (seq_len, seq_len)

    cos = np.cos(angles)
    sin = np.sin(angles)

    x_even = x[:, 0::2] # every second col starting from  0, simple
    x_odd = x[:, 1::2] # ditto but starting from 1

    rotated_even = (x_even * cos) - (x_odd * sin) # new_x0 = x0*cos(theta) - x1*sin(theta), simple rotation formula
    rotated_odd = (x_even * sin) + (x_odd * cos) # new_x1 = x0*sin(theta) + x1*cost(theta)

    output = np.empty_like(x)

    output[:, 0::2] = rotated_even
    output[:, 1::2] = rotated_odd

    return output