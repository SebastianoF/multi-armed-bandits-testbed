#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from setuptools import Command, find_packages, setup


def clean_requirements_list(input_list):
    reqs = [v.split("#")[0].strip() for v in input_list]
    return [v for v in reqs if len(v) > 0 and not v.startswith("-")]


class Cleaner(Command):
    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


with open("README.md") as f:
    readme = f.read()

with open("requirements/requirements.txt") as f:
    requirements = f.readlines()

with open("requirements/requirements-dev.txt") as f:
    requirements_dev = f.readlines()


requirements = clean_requirements_list(requirements)
requirements_dev = clean_requirements_list(requirements_dev)


setup(
    name="multi_armed_bandit",
    version="0.0.1",
    description="Multi armed bandit minimal implementation",
    long_description=readme,
    author="SebastianoF",
    author_email="sebastiano.ferraris@gmail.com",
    url="https://github.com/SebastianoF/multi-armed-bandits-testbed",
    packages=find_packages(include=["mab"], exclude=["docs", "laboratory", "examples"]),
    python_requires=">=3.6",
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
