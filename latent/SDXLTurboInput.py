from __future__ import annotations

import torch, numpy as np
from latent.latent import Latent
from models import ModelConfig
from diffusers import StableDiffusionXLPipeline

class SDXLTurboInput(Latent):
    def __init__(self, config:ModelConfig, a:torch.Tensor|None=None, b:torch.Tensor|None=None):
        super().__init__()
        self.config = config
        assert type(self.config.pipe) == StableDiffusionXLPipeline, f"SDXLTurboInput expected type of pipe to be StableDiffusionXLPipeline, got {type(self.config.pipe)}!"

        a_shape = (1, 77, 2048)
        b_shape = (1, 1280)

        a = torch.zeros(a_shape).cuda() if a is None else a
        b = torch.zeros(b_shape).cuda() if b is None else b

        assert a.shape == a_shape, f"SDXLTurboLatent's <a> vector is not the expected shape: {a_shape}"
        assert b.shape == b_shape, f"SDXLTurboLatent's <b> vector is not the expected shape: {b_shape}"
        
        self.a = a.detach()
        self.b = b.detach()
        self.result = None
    
    def repeat(self, n:int):
        self.a = self.a.repeat(n, 1, 1)
        self.b = self.b.repeat(n, 1)
        return self
    
    def __sub__(self, latent:SDXLTurboInput|float|int):
        assert type(latent) in [SDXLTurboInput, float, int], f"Cannot subtract SDXLTurboInput with unexpected: <{type(latent)}>"
        a_value, b_value = latent, latent
        if type(latent) == SDXLTurboInput:
            a_value = latent.a
            b_value = latent.b
        return SDXLTurboInput(self.config, self.a - a_value, self.b - b_value)

    def __add__(self, latent:SDXLTurboInput|float|int):
        assert type(latent) in [SDXLTurboInput, float, int], f"Cannot add SDXLTurboInput with unexpected: <{type(latent)}>"
        a_value, b_value = latent, latent
        if type(latent) == SDXLTurboInput:
            a_value = latent.a
            b_value = latent.b
        return SDXLTurboInput(self.config, self.a + a_value, self.b + b_value)

    def __len__(self):
        return self.a.shape[0]

    def push(self, latent:SDXLTurboInput):
        assert type(latent) == SDXLTurboInput, f"Cannot push SDXLTurboInput with unexpected: <{type(latent)}>"
        if len(self) == 1:
            self = latent
        else:
            self.a = torch.cat([self.a[1:].detach(), latent.a.detach().view(-1, 77, 2048)])
            self.b = torch.cat([self.b[1:].detach(), latent.b.detach().view(-1, 1280)])
    
    def from_prompt(self, prompt:str):
        a, _, b, _ = self.config.pipe.encode_prompt(prompt)
        return SDXLTurboInput(self.config, a, b)
    
    def dot(self, latent:SDXLTurboInput):
        # TODO, the more similar the two latents are (high dot), the smaller the exploit weight
        us = torch.cat([self.a.view(-1), self.b.view(-1)]).float()
        other = torch.cat([latent.a.view(-1), latent.b.view(-1)]).float()
        return torch.dot(us/(torch.norm(us) + 0.01), other/(torch.norm(other) + 0.01))
        

    def __truediv__(self, value:SDXLTurboInput|float|int):
        assert type(value) in [SDXLTurboInput, float, int], f"Cannot divide SDXLTurboInput with unexpected: <{type(value)}>"

        a_value, b_value = value, value
        if type(value) == SDXLTurboInput:
            a_value = torch.mean(value.a)
            b_value = torch.mean(value.b)
        
        a_value = 1 if a_value == 0 else a_value
        b_value = 1 if b_value == 0 else b_value
        
        return SDXLTurboInput(self.config, self.a / a_value, self.b / b_value)
    
    def __mul__(self, value:SDXLTurboInput|float|int):
        assert type(value) in [SDXLTurboInput, float, int], f"Cannot multiply SDXLTurboInput with unexpected: <{type(value)}>"

        if type(value) == SDXLTurboInput:
            a_value = value.a * self.a #torch.mean(value.a)
            b_value = value.b * self.b#torch.mean(value.b)
        else:
            a_value, b_value = self.a * value, self.b * value
        
        
        return SDXLTurboInput(self.config, a_value, b_value)

    def square(self):
        return SDXLTurboInput(self.config, self.a.square(), self.b.square())

    def sqrt(self):
        return SDXLTurboInput(self.config, self.a.sqrt(), self.b.sqrt())
    
    def inv(self):
        return SDXLTurboInput(self.config, 1 / self.a, 1/ self.b)
    
    def permute_mix(self, value:SDXLTurboInput, alpha:float=0.5):
        
        mask_a = torch.rand_like(self.a.view(-1)) < alpha
        mask_b = torch.rand_like(self.b.view(-1)) < alpha

        a_clone = self.a.clone().view(-1)
        b_clone = self.b.clone().view(-1)

        a_clone[mask_a] = value.a.view(-1)[mask_a]
        b_clone[mask_b] = value.b.view(-1)[mask_b]

        return SDXLTurboInput(self.config, a_clone.view(-1, 77, 2048), b_clone.view(-1, 1280))


    @property
    def norm(self):
        return SDXLTurboInput(
            self.config, 
            torch.zeros_like(self.a) + torch.norm(self.a), 
            torch.zeros_like(self.b) + torch.norm(self.b)
        )

    @property
    def unit(self):
        return self / self.norm

    def slerp(self, v0:SDXLTurboInput, v1:SDXLTurboInput, t:float):
        DOT_THRESHOLD = 0.9995

        dot = v0.dot(v1)
        if abs(dot) > DOT_THRESHOLD:
            v_out =  v0 * (1 - t) + v1 * t
        else:
            theta_0 = torch.arccos(dot)
            sin_theta_0 = torch.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = torch.sin(theta_t)
            s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v_out =  v0 * s0.item() + v1 * s1.item()
        
        return v_out

    @property
    def avg(self):
        # Take last 
        #return SDXLTurboInput(self.config, self.a[-1].view(1, 77, 2048), self.b[-1].view(1, 1280))

        # Simple Avg
        return SDXLTurboInput(self.config, torch.mean(self.a, dim=0).view(1, 77, 2048), torch.mean(self.b, dim=0).view(1, 1280))

        # Iter Slerp
        #slerped = [SDXLTurboInput(self.config, self.a[i].view(1, 77, 2048), self.b[i].view(1, 1280)) for i in range(len(self.a))]
        #while not len(slerped) == 1:            
        #    slerped.append(self.slerp(slerped[0], slerped[1], 0.5))
        #    slerped = slerped[2:]
        #return slerped[-1]
    
    def generate(self, positive:SDXLTurboInput|None=None, negative:SDXLTurboInput|None=None, seed:int=42):
        assert type(negative) == SDXLTurboInput or negative is None , f"Expected negative prompt to be of type SDXLTurboInput, got {type(negative)}!"
        assert type(positive) == SDXLTurboInput or positive is None, f"Expected positive prompt to be of type SDXLTurboInput, got {type(positive)}!"
        
        if not self.result is None:
            return self.result

        neg_a, neg_b = None, None
        if not negative is None: neg_a, neg_b = negative.a, negative.b

        pos_a, pos_b = None, None
        if not positive is None: pos_a, pos_b = positive.a, positive.b

        if seed: torch.manual_seed(seed)

        with torch.no_grad():
            self.result = self.config.pipe(

                # Prompt embeddings
                prompt_embeds = self.a + pos_a if not pos_a is None else self.a,
                pooled_prompt_embeds = self.b + pos_b if not pos_b is None else self.b, 
                negative_prompt_embeds = neg_a,
                negative_pooled_prompt_embeds = neg_b,
                
                # Inference settings
                num_inference_steps=self.config.steps,
                guidance_scale=self.config.guidance_scale,
                use_karras_sigmas=True,
                #euler_at_final=True,
                width=self.config.width,
                height=self.config.height,
                add_watermark=False
                

            ).images[0]

        torch.manual_seed(np.random.randint(1, 1e16))
        return self.result

    def raw_export(self):
        return (self.a, self.b)
        
    def raw_import(self, data):
        (self.a, self.b) = data