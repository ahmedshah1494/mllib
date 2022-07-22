from enum import Enum, auto
from time import time
from typing import List, Literal, Type, Union

from attrs import define
import attrs
from mllib.param import BaseParameters
import torch

from torchattacks.attack import Attack
import torchattacks
import foolbox

class SupportedAttacks(Enum):
    PGDL2 = auto()
    PGDLINF = auto()
    APGDLINF = auto()
    SQUARELINF = auto()
    RANDOMLY_TARGETED_SQUARELINF = auto()
    HOPSKIPJUMPLINF = auto()

class SupportedBackend(Enum):
    TORCHATTACKS = auto()
    FOOLBOX = auto()

@define(slots=False)
class AbstractAttackConfig:
    _cls = None

    def asdict(self):
        return attrs.asdict(self, filter=lambda attr, value: (not attr.name.startswith('_')))

def get_randomly_targeted_torchattack_cls(atkcls: torchattacks.attack.Attack):
    class RandomlyTargetedAttack(atkcls):
        __name__ = f'RandomlyTargeted{atkcls.__name__}'
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.num_classes = None
            self.set_mode_targeted_by_function(self.random_target_function)
        
        def random_target_function(self, x, y):
            return (y + torch.randint_like(y, self.num_classes - 1)) % self.num_classes
        
        def __call__(self, *input, **kwds):
            if self.num_classes is None:
                x = input[0].to(self.device)
                out = self.model(x)
                self.num_classes = out.shape[-1]
            return super().__call__(*input, **kwds)
    
    return RandomlyTargetedAttack

@define(slots=False)
class TorchAttackPGDInfParams(AbstractAttackConfig):
    _cls = torchattacks.PGD
    eps: float = 8/255
    nsteps: int = 10
    step_size: float = eps / (nsteps/2)
    random_start: bool = True

    def asdict(self):
        d = super().asdict()
        d['steps'] = d.pop('nsteps')
        d['alpha'] = d.pop('step_size')
        return d

@define(slots=False)
class TorchAttackAPGDInfParams(AbstractAttackConfig):
    _cls = torchattacks.APGD
    eps: float = 8/255
    nsteps: int = 10
    seed: int = time()
    n_restarts: int = 1

    def asdict(self):
        d = super().asdict()
        d['steps'] = d.pop('nsteps')
        return d

@define(slots=False)
class TorchAttackSquareInfParams(AbstractAttackConfig):
    _cls = torchattacks.Square
    eps: float = 8/255
    n_queries: int = 1000
    seed: int = time()
    n_restarts: int = 1

@define(slots=False)
class TorchAttackRandomlyTargetedSquareInfParams(TorchAttackSquareInfParams):
    _cls = get_randomly_targeted_torchattack_cls(TorchAttackSquareInfParams._cls)

class FoolboxAttackWrapper:
    atkcls: Type[foolbox.attacks.base.Attack] = None
    def __init__(self, model, **kwargs) -> None:
        self.model = foolbox.PyTorchModel(model, bounds=(0, 1))
        print(kwargs)
        self.run_kwargs = kwargs.pop('run_params')
        self.attack = self.atkcls(**kwargs)
    
    def __call__(self, x, y):
        return self.attack(self.model, x, y, **(self.run_kwargs))

class FoolboxHopSkipJumpInfWrapper(FoolboxAttackWrapper):
    atkcls = foolbox.attacks.HopSkipJumpAttack
@define(slots=False)
class FoolboxCommonRunParams(AbstractAttackConfig):
    epsilons: List[int] = [8/255]

@define(slots=False)
class FoolboxHopSkipJumpInfInitParams(AbstractAttackConfig):
    _cls = FoolboxHopSkipJumpInfWrapper
    steps: int = 64
    initial_gradient_eval_steps: int = 100
    max_gradient_eval_steps: int = 10000
    stepsize_search: Union[
        Literal["geometric_progression"], Literal["grid_search"]
    ] = "geometric_progression"
    gamma: float = 1.0
    tensorboard: Union[Literal[False], None, str] = False
    constraint: Union[Literal["linf"], Literal["l2"]] = "l2"
    run_params: AbstractAttackConfig = FoolboxCommonRunParams()

class AttackParamFactory:
    torchattack_params = {
        SupportedAttacks.PGDLINF: TorchAttackPGDInfParams,
        SupportedAttacks.APGDLINF: TorchAttackAPGDInfParams,
        SupportedAttacks.SQUARELINF: TorchAttackSquareInfParams,
        SupportedAttacks.RANDOMLY_TARGETED_SQUARELINF: TorchAttackRandomlyTargetedSquareInfParams
    }
    foolbox_params = {
        SupportedAttacks.HOPSKIPJUMPLINF: FoolboxHopSkipJumpInfInitParams
    }
    backend_params = {
        SupportedBackend.TORCHATTACKS: torchattack_params,
        SupportedBackend.FOOLBOX: foolbox_params
    }

    @classmethod
    def get_attack_params(self, attack: Type[SupportedAttacks], backend: Type[SupportedBackend]):
        return self.backend_params[backend][attack]()