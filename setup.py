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
        "numpy",
        'scipy >= 1',
        "networkx >= 2",
        "soothsayer_utils >= 2020.07.01",
        "hive_networkx >= 2020.06.30",
        "compositional >= 2020.05.19",
        # "matplotlib >= 3",
      ],
)
