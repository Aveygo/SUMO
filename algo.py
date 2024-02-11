from latent import Latent
from optimizer import AdamOptimizer

import random, time
from threading import Thread

#from realesr import Upscale

NEGATIVE_PROMPT = "worst quality, low quality, blurry, noisy, noise, distorted face, multiple limbs"

class QueuedImageGen(Thread):
    def __init__(self, profile:AdamOptimizer):
        Thread.__init__(self)

        # Number of pre-generated images, 
        # Does not prevent against user requesting images faster than generation
        self.q_size = 2

        self.profile:AdamOptimizer = profile
        
        # Latent that the user needs to pull
        self.to_pull = []

        # Latents that the user has pulled but not pushed
        self.floating = {}

        # The most recently pushed latent
        self.last_push_id = None

        # Negative embeddings for image generation
        self.negative = self.profile.from_prompt(NEGATIVE_PROMPT)

        # Upscaler was slower than using a larger width/height for diffusion
        #self.up = Upscale()

        self.start()
    
    def run(self):
        while True:
            while len(self.to_pull) >= self.q_size:
                time.sleep(0.01)

            line = random.choice(self.profile.config.lines)
            #print(line)      
            prompt = self.profile.from_prompt(line)
            aug_prompt = self.profile.position.slerp(self.profile.position, prompt, 1-self.profile.config.exploit_scale)
            
            aug_prompt.generate(negative=self.negative)
            self.to_pull.append(aug_prompt)

    def floating_cleanup(self):
        # Prevent spam by cleaning "floating" embeddings
        while len(self.floating) > 10:
            
            print(f"[WARNING] Bad client! Too many ({len(self.floating)}) floating latents!")

            # Removing oldest embeddings
            oldest_key = list(self.floating.keys())[0]
            oldest_time = self.floating[oldest_key]._sent

            for key in self.floating:
                if self.floating[key]._sent < oldest_time:
                    oldest_key = key
                    oldest_time = self.floating[key]._sent

            del self.floating[key]

    def pull(self, image_id:int) -> Latent:
        # User REQUESTS an image from the queue

        while len(self.to_pull) == 0:
            time.sleep(0.01)
        
        self.floating[image_id] = self.to_pull.pop(0)
        self.floating[image_id]._sent = time.time()

        return self.floating[image_id]
    
    def push(self, image_id:int) -> bool:
        # User is DONE viewing an image
        
        if not image_id in self.floating:
            return False

        if self.last_push_id is None:
            self.last_push_id = image_id
            return True
        
        if not self.last_push_id in self.floating:
            self.last_push_id = image_id
            return False
        
        self.profile.push(
            self.floating[self.last_push_id],
            self.floating[image_id],
            self.floating[image_id]._sent - self.floating[self.last_push_id]._sent,
            time.time() - self.floating[image_id]._sent,
        )
        del self.floating[self.last_push_id]
        self.last_push_id = image_id
        
        return True
