---
layout: post
title: "Post title"
summary: "One sentence shown in the post list and in search results."
math: true
---

<!--
HOW TO PUBLISH
1. Move this file from _drafts/ to _posts/.
2. Rename it to YYYY-MM-DD-your-slug.md (the date becomes the post date).
3. Commit. The post appears at /blog/YYYY/your-slug/.
-->

Opening paragraph: what the reader will understand by the end, in two or three sentences.

## Math

Inline math uses double dollars inside a sentence: $$f \sim \mathcal{GP}(m, k)$$. Display math is double dollars on their own lines:

$$
k(x, x') = \sigma^2 \exp\left(-\frac{\lVert x - x' \rVert^2}{2\ell^2}\right)
$$

$$
\mu_* = K_{*X} (K_{XX} + \sigma_n^2 I)^{-1} y
$$

## Code

```python
import numpy as np

def rbf(x1, x2, ell=1.0, sigma=1.0):
    d = x1[:, None] - x2[None, :]
    return sigma**2 * np.exp(-0.5 * d**2 / ell**2)
```

## Figures

Put images in `assets/img/` and reference them like this:

![Short description of the figure](/assets/img/your-figure.png)

## Takeaways

- One line per point the reader should remember.
