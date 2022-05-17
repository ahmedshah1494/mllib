from typing import Iterable, List, Type
from attrs import define
from torch import nn
import numpy as np
import torch
from param import BaseParameters

from utils.config import ConfigBase

class AbstractModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name: str = ''

    def compute_loss(self, x, y, return_logits=True):
        pass
    
    def checkpoint(self):
        def cast(x):
            if isinstance(x, Iterable):
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                    x = [cast(x_) for x_ in x]
                elif isinstance(x, list):
                    x = [cast(x_) for x_ in x]
                elif isinstance(x, dict):
                    x = {k:cast(v) for k,v in x.items()}
            elif not (isinstance(x, int) and isinstance(x, float)):
                x = str(x)
            assert any([isinstance(x, t) for t in [str, int, float, dict, list]])
            return x
                
        sd = self.state_dict()
        attributes = vars(self)
        attributes = {k:cast(v) for k,v in attributes.items()}
        return {'state_dict':sd, 'attributes':attributes}

class MLP(AbstractModel):
    # @define(slots=False)
    class MLPParams(BaseParameters):
        input_size: Iterable = 0
        widths: List[int] = [0]
        output_size: int = 0
        activation: Type[nn.Module] = nn.ReLU
        bias: bool = True
        add_shortcut_connections: bool = False                    
        block_boundaries: List[int] = []
        dropout_p: float = 0
        use_batch_norm: bool = False

    @classmethod
    def get_params(cls):
        return cls.MLPParams(cls)

    def __init__(self, params: MLPParams) -> None:
        super(MLP, self).__init__()
        self.name = f"MLP-{'_'.join([str(w) for w in params.widths])}"
        activation_str = params.activation().__str__()
        self.name +=f'-{activation_str[:activation_str.index("(")]}'
        self.dropout_p = params.dropout_p
        self.use_batch_norm = params.use_batch_norm
        if params.use_batch_norm:
            self.name += 'BN'
        if params.dropout_p > 0:
            self.name += f'Dropout{params.dropout_p}'
        self.add_shortcut_connections = params.add_shortcut_connections
        self.block_boundaries = params.block_boundaries if len(params.block_boundaries) > 0 else [len(params.widths)]
        if params.add_shortcut_connections:
            self.name += f'-wSkip_{">".join([str(x) for x in self.block_boundaries])}'
        self.input_size = int(np.prod(params.input_size))
        self.widths = params.widths
        self.output_size = params.output_size
        self.num_classes = params.output_size
        self.activation = params.activation
        self.use_bias = params.bias
        if not params.bias:
            self.name += '-noBias'
        self._initialize_mlp()
    
    def _get_activation(self, indim):
        acts = []
        if self.dropout_p > 0:
            acts.append(nn.Dropout(self.dropout_p))
        if self.use_batch_norm:
            acts.append(nn.BatchNorm1d(indim))
        acts.append(self.activation())
        return nn.Sequential(*acts)

    def _initialize_mlp(self):
        layers = []
        widths = [self.input_size, *self.widths]
        block_boundaries = np.array(self.block_boundaries) + 1
        bs = 1
        for i,be in enumerate(block_boundaries):
            block = []
            for j, w in enumerate(widths[bs:be]):
                if i > 0:
                    block.append(self._get_activation(widths[bs+j-1]))
                block.append(nn.Linear(widths[bs+j-1], w, bias=self.use_bias))
                if (bs+j < (be-1)):
                    block.append(self._get_activation(w))
            bs = be
            block = nn.Sequential(*block)
            layers.append(block)
        if len(widths) == 1:
            act = nn.Identity()
            w = self.input_size
        else:
            act = self._get_activation(w)
        layers.append(
            nn.Sequential(
                act,
                nn.Linear(w, self.output_size if self.output_size > 2 else 1, bias=self.use_bias)
            )
        )
        self.mlp = nn.ModuleList(layers)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for i, l in enumerate(self.mlp):
            if (i == 0) or (i == len(self.mlp)-1) or (not self.add_shortcut_connections):
                x = l(x)
                # print(f'x = {l}(x)')
            else:
                x = l(x) + x
                # print(f'x = {l}(x) + x')
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        pass

class MLPClassifier(MLP):
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        if self.output_size == 2:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss

class MLPRegressor(MLP):
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.mse_loss(x, y, reduction='sum') / x.shape[0]
        if return_logits:
            return logits, loss
        else:
            return loss

class MLPAutoRegressor(MLPRegressor):
    def compute_loss(self, x, y, return_logits=True):
        return super().compute_loss(x, x.view(x.shape[0], -1), return_logits)

class ConvConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.input_size: Iterable = None
        self.kernel_sizes: List[int] = None
        self.strides: List[int] = None
        self.padding: List[int] = None
        self.num_filters: List[int] = None
        self.activation=nn.ReLU
        self.use_bias: bool = True
        self.conv_class = nn.Conv2d
        self.dropout_p: float = 0.

class ConvEncoder(AbstractModel):
    def __init__(self, config: ConvConfig) -> None:
        super(ConvEncoder, self).__init__()
        
        layer_desc = [f'{f}x{ks}_{s}_{p}' for ks,s,p,f in zip(config.kernel_sizes, config.strides, config.padding, config.num_filters)]
        layer_desc_2 = []
        curr_desc = ''
        for i,x in enumerate(layer_desc):
            if x == curr_desc:
                count += 1
            else:
                if curr_desc != '':
                    layer_desc_2.append(f'{count}x{curr_desc}')
                count = 1
                curr_desc = x
            if i == len(layer_desc)-1:
                layer_desc_2.append(f'{count}x{curr_desc}')
        self.conv_class = config.conv_class
        self.name = 'Conv-'+'_'.join(layer_desc_2)
        self.input_size = config.input_size
        self.widths = []
        self.in_channels = config.input_size[0]
        self.num_filters = config.num_filters
        self.kernel_sizes = config.kernel_sizes
        self.strides = config.strides
        self.padding = config.padding
        self.activation = config.activation
        self.use_bias = config.use_bias
        self.dropout_p = config.dropout_p
        self._initialize_conv()
    
    def _initialize_conv(self):
        layers = []
        nfilters = [self.in_channels, *self.num_filters]
        kernel_sizes = [None] + self.kernel_sizes
        strides = [None] + self.strides
        padding = [None] + self.padding
        for i, (k,s,f,p) in enumerate(zip(kernel_sizes, strides, nfilters, padding)):
            if i > 0:
                layers.append(self.conv_class(nfilters[i-1], f, k, s, p, bias=self.use_bias))
                layers.append(self.activation())
                if self.dropout_p > 0:
                    layers.append(nn.Dropout2d(self.dropout_p))
        self.conv_model = nn.Sequential(*layers)
    
    def forward(self, x, return_state_hist=False, **kwargs):
        feat = self.conv_model(x)
        return feat
    
    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        logits = self.forward(x, return_state_hist=True)
        loss = torch.tensor(0., device=x.device)
        output = (loss,)
        if return_state_hist:
            output = output + (None,)
        if return_logits:
            output = (logits,) + output
        return output

class ConvAutoEncoder(ConvEncoder):
    def __init__(self, config: ConvConfig) -> None:
        super().__init__(config)
        self.name = self.name.replace('Conv', 'ConvAE')
        self.num_classes = 3

    def _initialize_conv(self):
        super()._initialize_conv()
        x = torch.rand((1,*(self.input_size)))
        layer_output_shapes = []
        for l in self.conv_model:
            x = l(x)
            layer_output_shapes.append(x.shape)
        
        num_filters = [self.input_size[0]] + self.num_filters
        conv_shapes = [s for l,s in zip(self.conv_model, layer_output_shapes) if isinstance(l, nn.Conv2d)]
        output_shapes = [self.input_size[1:]] + conv_shapes
        recon_layers = []
        for i in range(len(num_filters)-1):
            ic = num_filters[i+1]
            oc = num_filters[i]
            k = self.kernel_sizes[i]
            s = self.strides[i]
            p = self.padding[i]
            d_in = np.array(output_shapes[i+1][-2:])
            d_out = np.array(output_shapes[i][-2:])
            op = d_out - ((d_in - 1) * np.array(s) - 2 * np.array(p) + np.array(k)-1 + 1)
            l = nn.ConvTranspose2d(ic, oc, k, s, p, tuple(op))
            if i == 0:
                recon_layers.append(nn.Sigmoid())
            recon_layers.append(l)
            if i < len(num_filters) - 2:
                recon_layers.append(self.activation())
        self.reconstructor = nn.Sequential(*(recon_layers[::-1]))
    
    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        feat = self.forward(x)
        x_ = self.reconstructor(feat)
        loss = 0.5 * ((x_ - x)**2).reshape(x.shape[0], -1).sum(-1).mean(0)
        if return_logits:
            return feat, loss
        else:
            return loss


class ConvClassifier(ConvEncoder):
    def __init__(self, config: ConvConfig) -> None:
        super().__init__(config)
        self.linear_class = linear_class
        self.num_classes = num_classes
        self._initialize_linear()

    def _initialize_linear(self):
        x = torch.rand((1,*(self.input_size)))
        if len(x.shape) < 4:
            assert len(x.shape) == 3
            x = x.unsqueeze(1)

        self.layer_output_shapes = []
        for l in self.conv_model:
            x = l(x)
            self.layer_output_shapes.append(x.shape)
        
        layers = []
        num_logits = self.num_classes
        widths = [np.prod(x.shape[1:]), *self.widths, num_logits]
        for i, w in enumerate(widths):
            if i > 0:
                layers.append(self.linear_class(widths[i-1], w))
                if i < len(widths)-1:
                    layers.append(self.activation())
        self.linear_model = nn.Sequential(*layers)
    
    def forward(self, x, return_state_hist=False, **kwargs):
        feat = self.conv_model(x)
        logits = self.linear_model(feat.reshape(feat.shape[0], -1))
        if return_state_hist:
            return logits, feat.unsqueeze(1)
        return logits

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        if self.num_classes == 2:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss