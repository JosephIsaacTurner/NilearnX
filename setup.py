from setuptools import setup, find_packages

setup(
    name='nilearnx',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'nilearn',
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'seaborn',
        'nibabel',
        'joblib',
        'nitime',
        'statsmodels'
    ]
)