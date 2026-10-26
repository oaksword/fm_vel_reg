# Circle Jacobian diagnostics

## Diagnostic overview

Population-level numerical checks accompanying the paper *Geometric Velocity Regularity for Flow Matching on Manifold-Concentrated Data*. The script evaluates the flow-matching velocity Jacobian on the unit circle through the exact posterior-covariance identity. No velocity field is trained.

The source is uniform on $S^1$ by default, with one nonuniform density $p(\omega)\propto 1+0.5\cos\omega$. For $\sigma\in\{0.2,0.1,0.05,0.025,0.0125\}$ at the switching time $t^\star=1/(1+\sigma)$, the diagnostics are:

- centerline normal and tangential forms, and the operator norm, for the uniform circle
- tangential, normal, and mixed blocks for the nonuniform density
- time integrals of the centerline operator norm against the $\sigma_t^{-1}$ and $\sigma_t^{-3}$ envelopes
- operator-norm quantiles on held-out samples from the late marginal
- quadrature checked against the von Mises closed form

Running `python3 run_circle_jacobian.py` rewrites `results/`, including the CSV tables, `results/summary.md`, and the PDF figures. Dependencies are listed in `requirements.txt`.

## License

The code and numerical outputs in this repository are released under the MIT License.

```
Copyright (c) 2026 oaksword

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Third-party licenses

The diagnostic uses the following packages and no external datasets or pretrained models. `requirements.txt` specifies NumPy 2.0 or later and leaves the others unpinned.

- NumPy is under the BSD-3-Clause license. Some bundled components are under 0BSD, MIT, Zlib, and CC0-1.0. https://github.com/numpy/numpy/blob/main/LICENSE.txt
- SciPy is under the BSD-3-Clause license. https://github.com/scipy/scipy/blob/main/LICENSE.txt
- pandas is under the BSD-3-Clause license. https://github.com/pandas-dev/pandas/blob/main/LICENSE
- Matplotlib is under the Matplotlib license, which is based on the Python Software Foundation license and is BSD-compatible. https://matplotlib.org/stable/project/license.html
