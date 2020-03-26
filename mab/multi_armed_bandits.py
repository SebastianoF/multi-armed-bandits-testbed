import numpy as np
import attr

from attr.validators import instance_of

@attr
class MultiArmedBandit:
    K = attr.ib(default=10, validator=instance_of(int))
    mu = attr.ib(default=np.random., validator=instance_of(np.ndarray))
    # sigma =
