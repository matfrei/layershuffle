from setuptools import find_packages, setup

setup(
        name='layershuffle',
        packages=find_packages(),
        version='0.0.0.1',
        description='LayerShuffle: Enhancing Robustness in Vision Transformers by Randomizing Layer Execution Order',
        author='Anonymus',
        license='',
        install_requires=[
            "torch==2.2.1",
            "torchvision==0.17.1",
            "numpy==1.26.4",
            "pandas==2.2.1",
            "yacs",
            "transformers==4.38.2",
            "accelerate==0.27.2"])
