<div align="left">
  <img src="assets/images/PyCMF-logo.jpg" height="90px" alt="PyCMF Logo"/>
</div>

# PyCMF

PyCMF is a Python package including correlated mean-field methods for molecules and materials. It is built on top of the PySCF library.

## Installation

### Step 1: Clone the Repository

Download to your local machine

```bash
git clone https://github.com/QuantumLab-HCMUS/pyCMF.git
cd pyCMF

```

*(Make sure you are inside the `pyCMF` root directory where the `pyproject.toml` file is located before proceeding to the next steps).*

### Step 2: Create a Minimal Conda Environment

It is highly recommended to isolate the dependencies of this project. Create a clean Conda environment with Python 3.11:

```bash
conda create -n pycmf python=3.11 -y

```

### Step 3: Activate the Environment

You must activate the environment before installing anything:

```bash
conda activate pycmf

```

*(You should see `(pycmf)` appear at the beginning of your terminal prompt).*

### Step 4: Install the Package in Editable Mode

Install pyCMF along with all its core dependencies (pyscf, opt_einsum, numpy, scipy) AND development tools (ruff, pre-commit).

Run the following command. Do not forget the .[dev] at the end!

```Bash
pip install -e ".[dev]"

```

*(Note: The [dev] flag tells pip to look into the pyproject.toml file and install the necessary formatting and linting tools required for QuantumLab contributors).*

### Step 5: Initialize Pre-commit Hooks (For Contributors)

To maintain a clean and standard codebase, QuantumLab uses automated formatting (Ruff). You must activate the pre-commit hook before writing any code:

```Bash
pre-commit install

```

*(You should see a success message like pre-commit installed at .git/hooks/pre-commit. From now on, your code will be automatically formatted whenever you run git commit).*