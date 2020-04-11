# Template:
[![CircleCI](https://circleci.com/gh/SebastianoF/multi-armed-bandits-testbed.svg?style=svg)](https://app.circleci.com/pipelines/github/SebastianoF/multi-armed-bandits-testbed)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Output sample](https://github.com/SebastianoF/multi-armed-bandits-testbed/blob/master/docs/figures/sequence.gif)


# Multi armed bandit

An implementation of a range of epsilon-greedy algorithms to solve the multi-armed bandits problem.

### Install

* Clone the repo locally:
```bash
git clone git@github.com:SebastianoF/multi-armed-bandits-testbed.git
cd multi-armed-bandits-testbed
```
* Create and activate a [virtual environment](https://docs.python.org/3/tutorial/venv.html) (python 3.8):
```bash
virtualenv -p python3.8 venv
source venv/bin/activate
```

* Install in [development mode](https://flamy.ca/blog/2017-01-02-installing-python-packages-in-development-mode.html):
```bash
pip install -e .
```

### Where to start

Check out the examples in the folder [/examples/start_here.py](https://github.com/SebastianoF/multi-armed-bandits-testbed/blob/master/examples/start_here.py).

### Development

+ Dependent libraries are managed with pip-compile-multi
+ Continuous integration is integrated with CircleCI
+ Text formatting happens via pre-commit

### Resources

* R. Sutton, A. Barto, "Reinforcement Learning, an introduction", Chapter 1.

* Further introduction can be found in the folder [`/docs/`](https://github.com/SebastianoF/multi-armed-bandits-testbed/blob/master/docs/bourbaki_pragmatist_MAB.pdf)

### Licence

Repository open-sourced under [MIT](https://choosealicense.com/licenses/mit/) licence.
