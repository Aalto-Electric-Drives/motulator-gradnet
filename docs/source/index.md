# *motulator-gradnet:* Gradient-Network Magnetic Models for *motulator*

This package trains gradient networks (GradNets) that model the current and flux linkage maps of synchronous machines, including spatial harmonics, from measured or finite-element data {cite}`Li2026`. A GradNet is the gradient of a scalar state function, the magnetic energy or co-energy, so the map is a conservative vector field by construction {cite}`Cha2025`, and the monotonicity of the flux-linkage–current map is ensured. The networks are defined and trained with PyTorch. A trained network is saved as a NumPy function, which the magnetic maps of [*motulator*](https://aalto-electric-drives.github.io/motulator/) evaluate in the machine models and control systems without PyTorch. The package also compares the trained maps with the data.

This documentation covers the training. The magnetic maps and the machine models are described in the [documentation of *motulator*](https://aalto-electric-drives.github.io/motulator/model/drive/synchronous_machine.html#saturation-models), which also has simulation examples using the trained maps.

```{note}
The package is in early development, and its interfaces may change before version 1.0. It currently supports *motulator* 0.15, and the support is extended after each new release of *motulator* has been tested.
```

```{toctree}
---
titlesonly:
caption: Getting Started
name: getting_started
maxdepth: 1
---
installation
usage
```

```{toctree}
---
titlesonly:
caption: Examples
name: examples
maxdepth: 1
---
examples/index
```

```{toctree}
---
titlesonly:
caption: Reference
name: reference
maxdepth: 1
---
API <autoapi/motulator_gradnet/index>
bibliography
motulator <https://aalto-electric-drives.github.io/motulator/>
```
