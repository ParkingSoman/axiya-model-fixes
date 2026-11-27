#!/usr/bin/env python
"""Draw the test page used by before_after.ipynb: a page of text and 12 numbered formulas.

The page is drawn from scratch with matplotlib (mathtext), so it is free to share.
Output: page.png, 1654 x 2339 pixels (A4 at 200 dpi, a common scan size).

  python make_page.py            # writes page.png next to this script
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The 12 formulas on the page, in LaTeX. before_after.ipynb uses this list as the answer key.
FORMULAS = [
    r"\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}",
    r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
    r"\Gamma(z) = \int_0^\infty t^{z-1} e^{-t}\,dt",
    r"\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x)\, e^{-2\pi i x \xi}\,dx",
    r"G(\chi) = \sum_{a=1}^{p} \chi(a)\, e^{2\pi i a/p}",
    r"|G(\chi)|^2 = p",
    r"\zeta(s) = \prod_{p} \frac{1}{1 - p^{-s}}",
    r"\binom{n}{k} = \frac{n!}{k!\,(n-k)!}",
    r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
    r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
    r"\left(\sum_{i=1}^{n} a_i b_i\right)^2 \leq \sum_{i=1}^{n} a_i^2 \sum_{i=1}^{n} b_i^2",
]

# One line of text before each formula.
TEXT = [
    "We start with the Gaussian integral, which appears in every part of this note.",
    "Euler's value for the sum of the inverse squares is the next classical result.",
    "The Gamma function extends the factorial to complex arguments.",
    "For an integrable function f we use the Fourier transform in the form",
    "Let p be an odd prime and let chi be a nontrivial character mod p. The Gauss sum is",
    "A short computation with the character group gives the absolute value",
    "The Euler product links the zeta function to the primes.",
    "The binomial coefficient counts the subsets of size k of a set of size n.",
    "The roots of a quadratic equation are given by the familiar formula",
    "The following limit is used in the proof of Lemma 2.",
    "In physics, Gauss's law has the differential form",
    "Finally, the Cauchy-Schwarz inequality bounds the inner product.",
]

W, H, DPI = 1654, 2339, 200


def draw(path):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Notes on Gauss Sums and Classical Integrals", ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.935, "Section 1. Preliminaries", ha="center", fontsize=11, style="italic")
    y = 0.905
    step = 0.0715
    for i, (t, f) in enumerate(zip(TEXT, FORMULAS), 1):
        fig.text(0.1, y, t, fontsize=9.5)
        fig.text(0.5, y - 0.033, f"${f}$", ha="center", va="center", fontsize=12)
        fig.text(0.9, y - 0.033, f"({i})", ha="right", va="center", fontsize=10)
        y -= step
    fig.text(0.5, 0.02, "1", ha="center", fontsize=9)
    fig.canvas.draw()
    from PIL import Image
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    img.save(path)  # plain RGB PNG, no extra metadata


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.png")
    draw(out)
    print("wrote page.png")
