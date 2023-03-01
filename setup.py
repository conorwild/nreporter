from setuptools import setup, find_packages

setup(
    name="nreporter",
    version="0.0.1",
    author="Conor J. Wild",
    author_email="conorwild@gmail.com",
    description="A utility for tracking sample sizes during chained Pandas operations",
    packages=find_packages(),
    install_requires=[
        'pandas', 'ipython', 'Jinja2', 'numpy'
    ],
)
