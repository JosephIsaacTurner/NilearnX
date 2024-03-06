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
    ],
    # Including data files in the package
    package_data={
        # If the atlas_data directory is directly inside the nilearnx package
        'nilearnx': ['atlas_data/*'],
    },
    # Use MANIFEST.in to include additional files in the source distribution
    include_package_data=True,
)
