from __future__ import annotations

import torch, numpy as np
from latent.latent import Latent
import models

class StableCascadeInput(Latent):
    def __init__(self, config:models.StableCascade, a:torch.Tensor|None=None, b:torch.Tensor|None=None):
        super().__init__()
        self.config = config
        assert type(self.config) == models.StableCascade, f"StableCascadeInput expected type of pipe to be StableDiffusionXLPipeline, got {type(self.config.pipe)}!"

        a_shape = (1, 77, 1280)
        b_shape = (1, 1, 1280)

        a = torch.zeros(a_shape).cuda() if a is None else a
        b = torch.zeros(b_shape).cuda() if b is None else b

        assert a.shape == a_shape, f"SDXLTurboLatent's <a> vector is not the expected shape: {a_shape}, got {a.shape}"
        assert b.shape == b_shape, f"SDXLTurboLatent's <b> vector is not the expected shape: {b_shape}, got {b.shape}"
        
        self.a = a.detach()
        self.b = b.detach()
        self.result = None
    
    def repeat(self, n:int):
        self.a = self.a.repeat(n, 1, 1)
        self.b = self.b.repeat(n, 1, 1)
        return self
    
    def __sub__(self, latent:StableCascadeInput|float|int):
        assert type(latent) in [StableCascadeInput, float, int], f"Cannot subtract StableCascadeInput with unexpected: <{type(latent)}>"
        a_value, b_value = latent, latent
        if type(latent) == StableCascadeInput:
            a_value = latent.a
            b_value = latent.b
        return StableCascadeInput(self.config, self.a - a_value, self.b - b_value)

    def __add__(self, latent:StableCascadeInput|float|int):
        assert type(latent) in [StableCascadeInput, float, int], f"Cannot add StableCascadeInput with unexpected: <{type(latent)}>"
        a_value, b_value = latent, latent
        if type(latent) == StableCascadeInput:
            a_value = latent.a
            b_value = latent.b
        return StableCascadeInput(self.config, self.a + a_value, self.b + b_value)

    def __len__(self):
        return self.a.shape[0]

    def push(self, latent:StableCascadeInput):
        assert type(latent) == StableCascadeInput, f"Cannot push StableCascadeInput with unexpected: <{type(latent)}>"
        if len(self) == 1:
            self = latent
        else:
            self.a = torch.cat([self.a[1:].detach(), latent.a.detach().view(-1, 77, 1280)])
            self.b = torch.cat([self.b[1:].detach(), latent.b.detach().view(-1, 1, 1280)])
    
    def from_prompt(self, prompt:str):
        #a, _, b, _ = self.config.pipe.encode_prompt(prompt)
        #return StableCascadeInput(self.config, a, b)

        a, b, _, _ = self.config.prior.encode_prompt(prompt=prompt, batch_size=1, num_images_per_prompt=1, do_classifier_free_guidance=False, device="cuda")
        return StableCascadeInput(self.config, a, b)

    def dot(self, latent:StableCascadeInput):
        # TODO, the more similar the two latents are (high dot), the smaller the exploit weight
        us = torch.cat([self.a.view(-1), self.b.view(-1)]).float()
        other = torch.cat([latent.a.view(-1), latent.b.view(-1)]).float()
        return torch.dot(us/(torch.norm(us) + 0.01), other/(torch.norm(other) + 0.01))
        

    def __truediv__(self, value:StableCascadeInput|float|int):
        assert type(value) in [StableCascadeInput, float, int], f"Cannot divide StableCascadeInput with unexpected: <{type(value)}>"

        a_value, b_value = value, value
        if type(value) == StableCascadeInput:
            a_value = torch.mean(value.a)
            b_value = torch.mean(value.b)
        
        a_value = 1 if a_value == 0 else a_value
        b_value = 1 if b_value == 0 else b_value
        
        return StableCascadeInput(self.config, self.a / a_value, self.b / b_value)
    
    def __mul__(self, value:StableCascadeInput|float|int):
        assert type(value) in [StableCascadeInput, float, int], f"Cannot multiply StableCascadeInput with unexpected: <{type(value)}>"

        if type(value) == StableCascadeInput:
            a_value = value.a * self.a #torch.mean(value.a)
            b_value = value.b * self.b#torch.mean(value.b)
        else:
            a_value, b_value = self.a * value, self.b * value
        
        
        return StableCascadeInput(self.config, a_value, b_value)

    def square(self):
        return StableCascadeInput(self.config, self.a.square(), self.b.square())

    def sqrt(self):
        return StableCascadeInput(self.config, self.a.sqrt(), self.b.sqrt())
    
    def inv(self):
        return StableCascadeInput(self.config, 1 / self.a, 1/ self.b)
    
    def permute_mix(self, value:StableCascadeInput, alpha:float=0.5):
        
        mask_a = torch.rand_like(self.a.view(-1)) < alpha
        mask_b = torch.rand_like(self.b.view(-1)) < alpha

        a_clone = self.a.clone().view(-1)
        b_clone = self.b.clone().view(-1)

        a_clone[mask_a] = value.a.view(-1)[mask_a]
        b_clone[mask_b] = value.b.view(-1)[mask_b]

        return StableCascadeInput(self.config, a_clone.view(-1, 77, 1280), b_clone.view(-1, 1, 1280))


    @property
    def norm(self):
        return StableCascadeInput(
            self.config, 
            torch.zeros_like(self.a) + torch.norm(self.a), 
            torch.zeros_like(self.b) + torch.norm(self.b)
        )

    @property
    def unit(self):
        return self / self.norm

    def slerp(self, v0:StableCascadeInput, v1:StableCascadeInput, t:float):
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
        #return StableCascadeInput(self.config, self.a[-1].view(1, 77, 2048), self.b[-1].view(1, 1280))

        # Simple Avg
        return StableCascadeInput(self.config, torch.mean(self.a, dim=0).view(1, 77, 1280), torch.mean(self.b, dim=0).view(1, 1, 1280))

        # Iter Slerp
        #slerped = [StableCascadeInput(self.config, self.a[i].view(1, 77, 2048), self.b[i].view(1, 1280)) for i in range(len(self.a))]
        #while not len(slerped) == 1:            
        #    slerped.append(self.slerp(slerped[0], slerped[1], 0.5))
        #    slerped = slerped[2:]
        #return slerped[-1]
    
    def generate(self, positive:StableCascadeInput|None=None, negative:StableCascadeInput|None=None, seed:int=42):
        assert type(negative) == StableCascadeInput or negative is None , f"Expected negative prompt to be of type StableCascadeInput, got {type(negative)}!"
        assert type(positive) == StableCascadeInput or positive is None, f"Expected positive prompt to be of type StableCascadeInput, got {type(positive)}!"
        
        if not self.result is None:
            return self.result

        neg_a, neg_b = None, None
        if not negative is None: neg_a, neg_b = negative.a, negative.b

        pos_a, pos_b = None, None
        if not positive is None: pos_a, pos_b = positive.a, positive.b

        if seed: torch.manual_seed(seed)

        with torch.no_grad():    
            prior_output = self.config.prior(
                #prompt="borris johnson in a bathtub of beans",
                prompt_embeds = self.a,
                prompt_embeds_pooled = self.b,
                #negative_prompt_embeds = neg_a,
                #negative_pooled_prompt_embeds = neg_b,

                height=self.config.height,
                width=self.config.width,
                #negative_prompt=negative_prompt,
                guidance_scale=self.config.guidance_scale,
                num_images_per_prompt=1,
                num_inference_steps=self.config.prior_steps
            )

            self.result = self.config.decoder(
                image_embeddings=prior_output.image_embeddings.half(),
                prompt="", # Not needed?
                #negative_prompt=negative_prompt,
                guidance_scale=1, 
                output_type="pil",
                num_inference_steps=self.config.decoder_steps
            ).images[0]


        torch.manual_seed(np.random.randint(1, 1e16))
        return self.result

    def raw_export(self):
        return (self.a, self.b)
        
    def raw_import(self, data):
        (self.a, self.b) = data