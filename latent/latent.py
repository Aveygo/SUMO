from __future__ import annotations

class Latent:
    def __init__(self): pass
    def repeat(self, n:int): raise Exception("Not defined")
    def from_prompt(self, prompt:str)->Latent: raise Exception("Not defined")
    def __sub__(self, latent:Latent|float|int)->Latent: raise Exception("Not defined")
    def __add__(self, latent:Latent|float|int)->Latent: raise Exception("Not defined")
    def __truediv__(self, value:Latent|float|int)->Latent: raise Exception("Not defined")
    def __mul__(self, value:Latent|float|int)->Latent: raise Exception("Not defined")
    def __len__(self): raise Exception("Not defined")
    def square(self)->Latent: raise Exception("Not defined")
    def sqrt(self)->Latent: raise Exception("Not defined")
    def inv(self)->Latent: raise Exception("Not defined")
    def dot(self, latent:Latent)->Latent: raise Exception("Not defined")
    def push(self, latent:Latent): raise Exception("Not defined")
    def generate(self, positive:Latent|None=None, negative:Latent|None=None, seed:int=0): raise Exception("Not defined")
    def permute_mix(self, value:Latent, alpha:float=0.5): raise Exception("Not defined")
    def slerp(self, v0:Latent, v1:Latent, t:float)->Latent: raise Exception("Not defined")
    def raw_export(self, filename): raise Exception("Not defined")
    def raw_import(self, filename): raise Exception("Not defined")

    @property
    def avg(self)->Latent: raise Exception("Not defined")
    @property
    def unit(self)->Latent: raise Exception("Not defined")
    @property
    def norm(self)->Latent: raise Exception("Not defined")
