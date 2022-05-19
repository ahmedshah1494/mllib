from enum import Enum, auto
from typing import Type

from attrs import define
import attrs
from mllib.param import BaseParameters

from torchattacks.attack import Attack
from torchattacks import PGD

class SupportedAttacks(Enum):
    PGDL2 = auto()
    PGDLINF = auto()

class SupportedBackend(Enum):
    TORCHATTACKS = auto()

@define(slots=False)
class AbstractAttackConfig:
    _cls = None

    def asdict(self):
        return attrs.asdict(self, filter=lambda attr, value: (not attr.name.startswith('_')))

@define(slots=False)
class TorchAttackPGDInfParams(AbstractAttackConfig):
    _cls = PGD
    eps: float = 8/255
    nsteps: int = 10
    step_size: float = eps / (nsteps/2)
    random_start: bool = True

    def asdict(self):
        d = super().asdict()
        d['steps'] = d.pop('nsteps')
        d['alpha'] = d.pop('step_size')
        return d

class AttackParamFactory:
    torchattack_params = {
        SupportedAttacks.PGDLINF: TorchAttackPGDInfParams
    }
    backend_params = {
        SupportedBackend.TORCHATTACKS: torchattack_params
    }

    @classmethod
    def get_attack_params(self, attack: Type[SupportedAttacks], backend: Type[SupportedBackend]):
        return self.backend_params[backend][attack]()