[![PyPI version](https://img.shields.io/pypi/v/robotools)](https://pypi.org/project/robotools)
[![pipeline](https://github.com/jubiotech/robotools/workflows/pipeline/badge.svg)](https://github.com/jubiotech/robotools/actions)
[![coverage](https://codecov.io/gh/jubiotech/robotools/branch/master/graph/badge.svg)](https://codecov.io/gh/jubiotech/robotools)
[![documentation](https://readthedocs.org/projects/robotools/badge/?version=latest)](https://robotools.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/358629210.svg)](https://zenodo.org/badge/latestdoi/358629210)

# `robotools`
This is a package for debugging and planning liquid handling operations, writing worklist files for the Tecan FreedomEVO platform on the fly.

You can visit the documentation at https://robotools.readthedocs.io, where the [notebooks](https://github.com/jubiotech/robotools/tree/master/notebooks)
are rendered next to auto-generated API documentation.

# Installation
`robotools` is available through [PyPI](https://pypi.org/project/robotools/):

```
pip install robotools
```

# Contributing
The easiest way to contribute is to report bugs by opening [Issues](https://github.com/JuBiotech/robotools/issues).

We apply automated code style normalization using `black`.
This is done with a `pre-commit`, which you can set up like this:
1. `pip install pre-commit`
2. `pre-commit install`
3. `pre-commit run --all`

Step 2.) makes sure that the `pre-commit` runs automatically before you make a commit.

Step 3.) runs it manually.

# Usage and Citing
`robotools` is licensed under the [GNU Affero General Public License v3.0](https://github.com/JuBiotech/robotools/blob/master/LICENSE).

When using `robotools` in your work, please cite the [corresponding software version](https://doi.org/10.5281/zenodo.4697605).

```bibtex
@software{robotools,
  author       = {Michael Osthege and
                  Laura Helleckes},
  title        = {JuBiotech/robotools: v1.3.0},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.3.0},
  doi          = {10.5281/zenodo.5745938},
  url          = {https://doi.org/10.5281/zenodo.5745938}
}
```

Head over to Zenodo to [generate a BibTeX citation](https://zenodo.org/badge/latestdoi/358629210) for the latest release.
