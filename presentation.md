---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
```

+++ {"slideshow": {"slide_type": "slide"}}

<center>

# NumPy for Biological Image Analysis

### BeBi 205, May 11th 2023

#### Ross Barnowski, [@rossbar](https://github.com/rossbar) on GitHub

</center>

+++ {"slideshow": {"slide_type": "slide"}}

# Why NumPy?

- `ndarray`: A generic, n-dimensional array data structure
  * Fundamental data structure underlying the *Scientific Python Ecosystem*

+++ {"slideshow": {"slide_type": "subslide"}}

<center>

## The Scientific Python Ecosystem

<img src="images/scientific_python_ecosystem.png" alt="scientific_python_ecosystem" width=40%/>

</center>

Image credit: Jarrod Millman et. al. - [Array programming with NumPy][numpy-paper]

[numpy-paper]: https://www.nature.com/articles/s41586-020-2649-2

+++ {"slideshow": {"slide_type": "slide"}}

# A Bit of History

+++ {"slideshow": {"slide_type": "fragment"}}

- **Mid 90's/Early 00's**: desire for high-performance numerical computation in
  Python culminates in the `Numeric` [(pdf)][numeric-manual] library.

[numeric-manual]: https://numpy.org/_downloads/768fa66c250a0335ad3a6a30fae48e34/numeric-manual.pdf

+++ {"slideshow": {"slide_type": "fragment"}}

- Early adopters included the [Space Telescope Science Institute (STScI)][stsci]
  who adapted Numeric to better suit their needs: `NumArray`.

[stsci]: http://www.stsci.edu/

+++ {"slideshow": {"slide_type": "fragment"}}

- **2005** The best ideas from `Numeric` and `NumArray` were combined in the
  development of a new library: `NumPy`
   * Originally `scipy.core` rather than a standalone library.
   * This work was largely done by [Travis Oliphant][travis-gh],
     then an assistant professor at BYU.

[travis-gh]: https://github.com/teoliphant

+++ {"slideshow": {"slide_type": "fragment"}}

- **2006** NumPy v1.0 released in October
