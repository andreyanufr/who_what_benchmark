from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='whowhatbench',
    version='1.0.0',
    url='https://github.com/andreyanufr/who_what_benchmark.git',
    author='Intel',
    author_email='',
    description='Short test for LLMs',
    packages=find_packages(),
    install_requires=required,
)
