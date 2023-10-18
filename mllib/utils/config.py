import argparse

class ConfigBase(object):    
    @classmethod
    def load(cls, src) -> None:
        valid_types = [dict, argparse.Namespace]
        config = cls()
        assert any([isinstance(src, t) for t in valid_types])
        if isinstance(src, argparse.Namespace):
            src = vars(src)
        for k,v in src.items():
            if isinstance(v, dict):
                c = ConfigBase.load(v)
                v = c
            setattr(config, k, v)
        return config

    def to_dict(self):
        d = vars(self)
        d = {k: v.to_dict() if hasattr(v, 'to_dict') else v for k,v in d.items()}
        return d
    
    def _null_config(self):
        self_vars = vars(self)
        for k,v in self_vars.items():
            if isinstance(v, ConfigBase):
                v._null_config()
            else:
                setattr(self, k, None)                

    def load_from_cmdline_args(self, src, add_missing_keys=False) -> None:
        self_vars = self.to_dict()
        valid_types = [dict, argparse.Namespace]
        assert any([isinstance(src, t) for t in valid_types])
        if isinstance(src, argparse.Namespace):
            src = vars(src)
        for k,v in src.items():
            if k in self_vars.keys():
                setattr(self, k, v)
            else:
                key_found = False
                for sk,sv in self_vars.items():
                    if isinstance(getattr(self, sk), ConfigBase) and (k in sv.keys()):
                        setattr(getattr(self, sk), k, v)
                        key_found = True
                if (not key_found) and add_missing_keys:
                    setattr(self, k, v)