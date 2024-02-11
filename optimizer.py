from typing import Type

from models import ModelConfig
from latent import Latent
import pickle, numpy as np


class AdamOptimizer:
    """
    Adam "gradient descent" 
    """
    def __init__(
            self,
            config:ModelConfig,
            latent_type:Type[Latent],
            lr=0.2,
            weight_decay=0.2
        ):
        self.config = config
        self.scores:list[float] = []
        self.lr = lr
        self.weight_decay = weight_decay

        self.position:Latent = latent_type(self.config)
        self.velocity:Latent = latent_type(self.config)
        self.momentum:Latent = latent_type(self.config)

    def calc_winner_bonus(self, score1, score2):
        delta_score = abs(score1 - score2)
        self.scores.append(delta_score)
        self.scores = self.scores[-16:]
        return float(1-2.71**(-1 * np.mean(self.scores) * delta_score) if len(self.scores) > 3 else 0.632)

    def push(self, l1:Latent, l2:Latent, score1:float, score2:float):
        winner_bonus = self.calc_winner_bonus(score1, score2) 
        selected, ignored = (l1, l2) if score1 > score2 else (l2, l1)

        # Sorry for the complicated gradient, it's just "selected - ignored" but in slerp space
        # I also "amplify" the gradient if the user spent alot of time on it.
        gradient:Latent = (self.position.slerp(ignored, selected, 2) - selected ) * winner_bonus * 2

        # Possible improvement? - Doesnt like to move?
        #candidates_gradient:Latent = self.position.slerp(ignored, selected, 2) - selected
        #position_gradient:Latent = self.position.slerp(self.position, selected, 2) - selected
        #gradient:Latent = (candidates_gradient + position_gradient) * winner_bonus

        self.momentum = self.momentum * self.weight_decay + gradient * (1 - self.weight_decay)
        self.velocity = self.velocity * self.weight_decay + gradient.square() * (1 - self.weight_decay)  
        self.position = self.position + (self.velocity.sqrt() + 1e-8).inv() * self.lr * self.momentum
        
    def from_prompt(self, prompt:str) -> Latent:
        return self.position.from_prompt(prompt)
    
    def save(self, dst:str='saved_latents/primary.pickle'):
        p = self.position.raw_export()
        v = self.velocity.raw_export()
        m = self.momentum.raw_export()

        with open(dst, 'wb') as f:
            pickle.dump((p, v, m), f, protocol=pickle.HIGHEST_PROTOCOL)

        return self

    def load(self, src:str='saved_latents/primary.pickle'):
        with open(src, 'rb') as f:
            p, v, m = pickle.load(f)
            self.position.raw_import(p)
            self.velocity.raw_import(v)
            self.momentum.raw_import(m)

        return self