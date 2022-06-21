from typing import Type
from attrs import define
import attrs
from copy import deepcopy


@define(slots=False)
class BaseParameters:
    cls: Type

    def __attrs_post_init__(self, *args, **kwargs):
        for k,v in vars(self).items():
            if isinstance(v, object):
                self.__setattr__(k, deepcopy(v))

    def asdict(self):
        return attrs.asdict(self)

class Parameterized:
    @classmethod
    def get_params(cls) -> Type[BaseParameters]:
        pass