from setuptools import setup, find_packages

setup(
    name='whowhatbench',
    version='1.0.0',
    url='https://github.com/andreyanufr/who_what_benchmark.git',
    author='Intel',
    author_email='',
    description='Short test for LLMs',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.35.2',
        'sentence-transformers>=2.2.2',
        'openvino-nightly==2023.3.0.dev20231113',
        'openvino-telemetry==2023.2.1',
        'optimum==1.14.1',
        'optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@f248835b16ce4ec054d6d4d629dff4213fe94157',
        'pandas>=2.0.3',
        'numpy>=1.23.5',
        'tqdm>=4.66.1'
    ],
)
