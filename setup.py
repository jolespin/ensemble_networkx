from setuptools import setup

# Version
version = None
with open("ensemble_networkx/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in ensemble_networkx/__init__.py"

setup(
name='ensemble_networkx',
    version=version,
    description='Ensemble networks in Python',
    url='https://github.com/jolespin/ensemble_networkx',
    author='Josh L. Espinoza',
    author_email='jespinoz@jcvi.org',
    license='BSD-3',
    packages=["ensemble_networkx"],
    install_requires=[
        "pandas >= 1",
        "numpy >= 1.16",
        'scipy >= 1',
        "networkx >= 2",
        "igraph",
        "xarray >= 0.15",
        "soothsayer_utils >= 2022.6.24",
        "compositional >= 2023.8.28",
        "scikit-learn >= 1.0",
      ],
)
