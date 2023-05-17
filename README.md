# NumPy for Biological Imaging Analysis

Presentation on NumPy prepared for BeBi 205, May 11th 2023

## Setup

Create a python virtual environment and enter it:

```bash
python -m venv bebi205-env
source bebi205-env/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## View the presentation

```bash
jupyter notebook presentation.md
```

**NOTE:**
- The jupyter notebook is in [MyST-markdown][myst] format and is automatically
  converted to .ipynb format with [`jupytext`][jupytext].
- The slideshow-mode is provided by the [`RISE`][rise] extension.

RISE quick start:

- Enter presentation mode with `alt+r`
- Toggle full-screen with `F11`
- Navigation:
  * Move one slide/fragment forward: `space`
  * Move one slide/fragment back: `shift+space`
- Run executable cells just like a normal notebook: `shift+enter`
