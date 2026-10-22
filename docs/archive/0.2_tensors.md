# 3. One tensor per population

Each population's activation layer says where that population's fibres are. Its tensor is the image's structure
tensor with every gradient weighted by the activation and by the boundary excision, integrated over the same
window ρ as the baseline and normalised by the integrated weight:

$$S_r = \frac{G_\rho * \left(a_r\, w\, \nabla I\, \nabla I^{\mathsf T}\right)}{G_\rho * (a_r\, w)}$$

Where the population is present its own fibres' gradients make the tensor, and only those; elsewhere the weight
is at the noise level. The eigenvector of the smallest eigenvalue is the population's direction, and the
orientation gap (λ₂ − λ₃) / λ₂ says how unique it is.

This is the like-for-like counterpart of the structure tensor: same derivatives, same window, same excision; the
only difference is the weight. On a single clean bundle it reproduces the structure tensor (2.6° against 3.1° on
the phantom); at a crossing each population's tensor sees its own fibres.

```python
from shapetensor.tensors import population_tensor
S, direction, gap = population_tensor(image, activation, sigma=1.0, rho=10.0, excision=weights)
```

The structure tensor of the activation image itself (`layer_tensor`) is the alternative; it is noisier, because
the activation is patchy at the fascicle scale, and the pipeline does not use it.
