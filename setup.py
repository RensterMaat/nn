from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn", "matplotlib"],
)
