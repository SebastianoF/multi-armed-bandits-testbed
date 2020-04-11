#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements
from setuptools import Command, find_packages, setup


class Cleaner(Command):
    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


with open("README.md") as f:
    readme = f.read()

requirements = [
    str(requirement).split()[0]
    for requirement in parse_requirements(
        "requirements/requirements.txt", session=PipSession()
    )
]

requirements_dev = [
    str(requirement).split()[0]
    for requirement in parse_requirements(
        "requirements/requirements.txt", session=PipSession()
    )
]


setup(
    name="multi_armed_bandit",
    version="0.0.1",
    description="Multi armed bandit minimal implementation",
    long_description=readme,
    author="SebastianoF",
    author_email="sebastiano.ferraris@gmail.com",
    url="https://github.com/SebastianoF/multi-armed-bandits-testbed",
    packages=find_packages(include=["mab"], exclude=["docs", "laboratory", "examples"]),
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    keywords="multi armed bandit",
    classifiers=[
        "Intended Audience :: Developers/Researchers",
        "Language :: English",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={"clean": Cleaner},
)
