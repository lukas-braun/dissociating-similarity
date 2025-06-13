# dissociation

Code to reproduce figures from our ICML 2025 paper "[Not all solutions are created equal: An analytical dissociation of functional and representational similarity in deep linear neural networks](#citation)."

## Setup

```bash
# If necessary, install UV.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project folder.
cd dissociation/

# Install dependencies.
uv sync

# Install the package.
uv pip install -p $(<.python-version) -e .
```

## Usage

```bash
# Install a Jupyter kernel for this project.
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=dissociation

# Explore notebooks to reproduce figures from the paper.
uv run --with jupyter jupyter lab --notebook-dir=notebooks
```

## Requirements

- Python 3.11+
- `uv` (https://astral.sh/uv/)

`uv` will take care of installing all dependencies, including Jupyter and the required Python packages, based on the `pyproject.toml` file.

## Citation

```bibtex
@inproceedings{braun2025dissociation,
    title={Not all solutions are created equal: An analytical dissociation of functional and representational similarity in deep linear neural networks},
    author={Braun, Lukas and Grant, Erin and Saxe, Andrew M.},
    booktitle={Proceedings of the 42nd International Conference on Machine Learning},
    year={2025},
    month={July},
    series={Proceedings of Machine Learning Research},
    publisher={PMLR},
    editor={Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Smith, Virginia and Berkenkamp, Felix and Maharaj, Tegan}
}
```
