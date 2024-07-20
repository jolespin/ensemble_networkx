from setuptools import setup

from os import path

script_directory = path.abspath(path.dirname(__file__))

# Version
package_name = "ensemble_networkx"
version = None
with open(path.join(script_directory, package_name, '__init__.py')) as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, f"Check version in {package_name}/__init__.py"


requirements = list()
with open(path.join(script_directory, 'requirements.txt')) as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            if not line.startswith("#"):
                requirements.append(line)
                
setup(
name='ensemble_networkx',
    version=version,
    description='Ensemble networks in Python',
    url='https://github.com/jolespin/ensemble_networkx',
    author='Josh L. Espinoza',
    author_email='jespinoz@jcvi.org',
    license='BSD-3',
    packages=["ensemble_networkx"],
    install_requires=requirements,
)
