from enum import Enum, auto
from time import time
from typing import List, Type, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from attrs import define, field
import attrs
from mllib.param import BaseParameters
import torch

from torchattacks.attack import Attack
import torchattacks
from autoattack.autopgd_base import APGDAttack
import foolbox

class SupportedAttacks(Enum):
    PGDL2 = auto()
    PGDLINF = auto()
    APGDLINF = auto()
    APGDL2 = auto()
    SQUARELINF = auto()
    RANDOMLY_TARGETED_SQUARELINF = auto()
    HOPSKIPJUMPLINF = auto()
    BOUNDARY = auto()
    CWL2 = auto()
    AUTOATTACK = auto()
    APGDL1 = auto()

class SupportedBackend(Enum):
    TORCHATTACKS = auto()
    AUTOATTACK = auto()
    FOOLBOX = auto()

@define(slots=False)
class AbstractAttackConfig:
    _cls = None
    model = None

    def asdict(self):
        return attrs.asdict(self, filter=lambda attr, value: (not attr.name.startswith('_')))

def get_randomly_targeted_torchattack_cls(atkcls: torchattacks.attack.Attack):
    class RandomlyTargetedAttack(atkcls):
        __name__ = f'RandomlyTargeted{atkcls.__name__}'
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.set_mode_targeted_random()

        @torch.no_grad()
        def _get_random_target_label(self, images, labels=None):
            seed = (images.detach().cpu().numpy().mean() * 1e5).astype(int)
            init_seed = torch.initial_seed()
            torch.manual_seed(seed)
            tgt = super()._get_random_target_label(images, labels)
            torch.manual_seed(init_seed)
            return tgt
        
        def __call__(self, *input, **kwds):
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
    norm: str = 'Linf'
    eps: float = 8/255
    nsteps: int = 100
    n_restarts: int = 1
    seed: int = field(factory=lambda : int(time()))
    loss: str = 'ce'
    eot_iter: int = 1
    rho: float = .75
    verbose:bool=False

    def asdict(self):
        d = super().asdict()
        d['steps'] = d.pop('nsteps')
        return d

@define(slots=False)
class TorchAttackAPGDL2Params(AbstractAttackConfig):
    _cls = torchattacks.APGD
    norm: str = 'L2'
    eps: float = 1.
    nsteps: int = 100
    n_restarts: int = 1
    seed: int = field(factory=lambda : int(time()))
    loss: str = 'ce'
    eot_iter: int = 1
    rho: float = .75
    verbose:bool=False

    def asdict(self):
        d = super().asdict()
        d['steps'] = d.pop('nsteps')
        return d

class AutoAttackkWrapper:
    atkcls = None
    def __init__(self, *args, **kwargs) -> None:
        self.attack = self.atkcls(*args, **kwargs)
    
    def __call__(self, x, y):
        return self.attack.perturb(x, y)

class WrappedAutoAttackAPGD(AutoAttackkWrapper):
    atkcls = APGDAttack

@define(slots=False)
class AutoAttackAPGDL1Params(AbstractAttackConfig):
    _cls = WrappedAutoAttackAPGD
    norm: str = 'L1'
    eps: float = 1.
    nsteps: int = 100
    n_restarts: int = 1
    seed: int = field(factory=lambda : int(time()))
    loss: str = 'ce'
    eot_iter: int = 1
    rho: float = .75
    verbose:bool=False

    def asdict(self):
        d = super().asdict()
        d['n_iter'] = d.pop('nsteps')
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

class AutoAttack(torchattacks.attack.Attack):
    def __init__(self, model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self._supported_mode = ['default']
        self.autoattack = torchattacks.MultiAttack([
            torchattacks.APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=1, steps=100),
            torchattacks.APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1, steps=100),
            torchattacks.FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1, steps=100),
            torchattacks.Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
        ])

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time() if self.seed is None else self.seed


@define(slots=False)
class TorchAttackAutoAttackParams(AbstractAttackConfig):
    _cls = AutoAttack
    eps: float = 8/255
    norm: str = 'Linf'
    version='standard'
    n_classes=10
    seed=None
    verbose=False

class FoolboxAttackWrapper:
    atkcls: Type[foolbox.attacks.base.Attack] = None
    def __init__(self, model, **kwargs) -> None:
        self.model = foolbox.PyTorchModel(model, bounds=(0, 1))
        print(kwargs)
        self.run_kwargs = kwargs.pop('run_params', {})
        self.attack = self.atkcls(**kwargs)
    
    def __call__(self, x, y):
        raw_advs, clipped_advs, success = self.attack(self.model, x, y, **(self.run_kwargs))
        return clipped_advs[0]

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
    run_params: FoolboxCommonRunParams = field(factory=FoolboxCommonRunParams)

class FoolboxBoundaryAttackWrapper(FoolboxAttackWrapper):
    atkcls = foolbox.attacks.BoundaryAttack
@define(slots=False)
class FoolboxBoundaryAttackInitParams(AbstractAttackConfig):
    _cls = FoolboxBoundaryAttackWrapper
    steps: int = 10
    spherical_step: float = 1e-2
    source_step: float = 1e-2
    source_step_convergance: float = 1e-7
    step_adaptation: float = 1.5
    update_stats_every_k: int = 10
    run_params: FoolboxCommonRunParams = field(factory=FoolboxCommonRunParams)

class FoolboxCWL2AttackWrapper(FoolboxAttackWrapper):
    atkcls = foolbox.attacks.L2CarliniWagnerAttack
@define(slots=False)
class FoolboxCWL2AttackInitParams(AbstractAttackConfig):
    _cls = FoolboxCWL2AttackWrapper
    binary_search_steps: int = 9
    steps: int = 10000
    stepsize: float = 1e-2
    confidence: float = 0
    initial_const: float = 1e-3
    abort_early: bool = True
    run_params: FoolboxCommonRunParams = field(factory=FoolboxCommonRunParams)

class AttackParamFactory:
    torchattack_params = {
        SupportedAttacks.PGDLINF: TorchAttackPGDInfParams,
        SupportedAttacks.APGDLINF: TorchAttackAPGDInfParams,
        SupportedAttacks.APGDL2: TorchAttackAPGDL2Params,
        SupportedAttacks.SQUARELINF: TorchAttackSquareInfParams,
        SupportedAttacks.RANDOMLY_TARGETED_SQUARELINF: TorchAttackRandomlyTargetedSquareInfParams,
        SupportedAttacks.AUTOATTACK: TorchAttackAutoAttackParams,
    }
    foolbox_params = {
        SupportedAttacks.HOPSKIPJUMPLINF: FoolboxHopSkipJumpInfInitParams,
        SupportedAttacks.BOUNDARY: FoolboxBoundaryAttackInitParams,
        SupportedAttacks.CWL2: FoolboxCWL2AttackInitParams,
    }
    autoattack_params = {
        SupportedAttacks.APGDL1: AutoAttackAPGDL1Params
    }
    backend_params = {
        SupportedBackend.TORCHATTACKS: torchattack_params,
        SupportedBackend.FOOLBOX: foolbox_params,
        SupportedBackend.AUTOATTACK: autoattack_params
    }

    @classmethod
    def get_attack_params(self, attack: Type[SupportedAttacks], backend: Type[SupportedBackend]):
        return self.backend_params[backend][attack]()