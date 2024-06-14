import logging
import os
import setuptools


def readlines(fname: str) -> list:
    """Load a local filename."""
    lines = open(os.path.join(os.path.dirname(__file__), fname)).readlines()
    return [line.strip() for line in lines]

def get_requirements():
    requirements = '\n'.join(readlines('requirements.txt'))
    return requirements
    

setuptools.setup(
    name="micron",
    version="0.2.0",
    author="Dmitry Karpeyev",
    author_email="dmitry.karpeyev@gmail.com",
    description="micro RNA LLM package",
    packages=setuptools.find_packages(exclude=("test,")),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={'console_scripts': []},
    python_requires='>=3.8',
    install_requires=get_requirements(),
)
