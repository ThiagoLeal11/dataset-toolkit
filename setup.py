from setuptools import setup, find_packages

setup(
    name='dataset_toolkit',
    description='A bunch of helpful scripts to make easier manage large datasets for DL training.',
    version='0.0.1',
    url='https://github.com/ThiagoLeal11/dataset-toolkit',
    author='Thiago L. Pozati',
    author_email='thiagoleal11@gmail.com',
    keywords=['dataset', 'ml', 'machine', 'learning', 'deep', 'deeplearning'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pillow',
        'numpy',
        'requests',
        'torch',
    ],
)
