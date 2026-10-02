# Matrix Factorization

## Model and Objective

For user `u` and item `i`, the model predicts the un-clipped rating

`r_hat_ui = mu + b_u + b_i + p_u dot q_i`,

where `mu` is the training-set mean, `b_u` and `b_i` are user and item biases,
and `p_u`, `q_i` are `f`-dimensional latent-factor vectors. For a mini-batch `B`
of `m` observed ratings, let `e_ui = r_hat_ui - r_ui`. The optimized loss is

`L_B = (1 / (2m)) sum_(u,i in B) e_ui^2`
`      + (lambda / (2m)) sum_(u,i in B) (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)`.

The regularizer is summed per observed rating, so repeated users and items
contribute once for each batch interaction. The global mean `mu` is fixed to the
training rating mean; it is not updated by SGD.

## Gradients

For each interaction, the prediction derivatives are

`d r_hat_ui / d p_u = q_i`,
`d r_hat_ui / d q_i = p_u`,
`d r_hat_ui / d b_u = 1`, and
`d r_hat_ui / d b_i = 1`.

Applying the chain rule to the batch loss gives

`d L_B / d p_u = (1 / m) sum_(i: (u,i) in B) (e_ui q_i + lambda p_u)`,
`d L_B / d q_i = (1 / m) sum_(u: (u,i) in B) (e_ui p_u + lambda q_i)`,
`d L_B / d b_u = (1 / m) sum_(i: (u,i) in B) (e_ui + lambda b_u)`,
`d L_B / d b_i = (1 / m) sum_(u: (u,i) in B) (e_ui + lambda b_i)`.

The sums account for all occurrences of a user or item in the batch. In code,
NumPy computes each interaction's vector contribution at once, and `numpy.add.at`
accumulates contributions for repeated users/items without a Python loop over
ratings.

## Mini-Batch SGD Updates

For learning rate `eta`, each accumulated gradient is applied as

`p_u <- p_u - eta * d L_B / d p_u`,
`q_i <- q_i - eta * d L_B / d q_i`,
`b_u <- b_u - eta * d L_B / d b_u`,
`b_i <- b_i - eta * d L_B / d b_i`.

Training predictions and losses use the un-clipped model output. Clipping to the
rating range `[1, 5]` is performed only by the public prediction method.