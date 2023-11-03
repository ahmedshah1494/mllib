from setuptools import setup, find_packages
from mllib import __version__

setup(
    name='mllib',
    version=__version__,

    url='https://github.com/ahmedshah1494/mllib',
    author='Muhammad Ahmed Shah',
    author_email='mshah1@cmu.edu',

    py_modules=find_packages(),

    install_requires=[
        'attrs==21.4.0',
        'numpy==1.22.3',
        'pillow==9.1.1',
        'scipy==1.7.3',
        'statsmodels==0.13.2',
        'torch==1.11.0',
        'torchaudio==0.11.0',
        'torchvision==0.12.0',
        'tqdm==4.64.0',
        'typing_extensions==4.2.0',
        'foolbox==3.3.3',
        'sentencepiece==0.1.97',
        'torchattacks==3.2.6',
        'torchmetrics==0.9.3',
        'webdataset==0.2.20',
        'tensorboard==2.10.0',
        'tensorboard-data-server==0.6.1',
        'tensorboardx==2.5.1',
        'pytorch-lightning==1.7.6'
    ],
)