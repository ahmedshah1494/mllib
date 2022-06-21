from typing import Type
from attrs import define
import attrs
from copy import deepcopy


@define(slots=False)
class BaseParameters:
    cls: Type

    def asdict(self):
        return attrs.asdict(self)

class Parameterized:
    @classmethod
    def get_params(cls) -> Type[BaseParameters]:
        pass