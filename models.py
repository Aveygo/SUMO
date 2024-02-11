from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
import torch

class ModelConfig:
    def __init__(self, seed_prompts:str=None):
        # Diffusers pipe to generate an image
        self.pipe = None
        
        # Size of output image
        self.width = 512
        self.height = 512

        # Steps for diffusion sampler, higher=longer(better)
        self.steps = 5

        # Scale for diffusion sampler, higher=closer to prompt
        self.guidance_scale = 2

        # Scale for profile features, higher=closer to predicted prompt
        self.exploit_scale = 1

        # prompts to create initial images
        with open(seed_prompts, "r") as f:
            self.lines = [i[:-1] for i in f.readlines()]

    def __repr__(self):
        return f"Steps: {self.steps}, Guidance: {self.guidance_scale}, Features: {self.exploit_scale}, Width: {self.width}, Height: {self.height}"

class StabilityTurbo(ModelConfig):
    def __init__(self, seed_prompts:str="seed_prompts/primary.txt"):
        super().__init__(seed_prompts)
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

        self.width = 512
        self.height = 512
        self.steps = 3
        self.guidance_scale = 1
        self.exploit_scale = 0.5

class RealVisTurbo(ModelConfig):
    def __init__(self, seed_prompts:str="seed_prompts/primary.txt"):
        super().__init__(seed_prompts)
        self.pipe = StableDiffusionXLPipeline.from_single_file("models/realvisxlV30Turbo_v30TurboBakedvae.safetensors", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

        self.width = 1024
        self.height = 1024
        self.steps = 6
        self.guidance_scale = 2
        self.exploit_scale = 0.3

class DreamShaperTurbo(ModelConfig):
    def __init__(self, seed_prompts:str="seed_prompts/primary.txt"):
        super().__init__(seed_prompts)
        self.pipe = StableDiffusionXLPipeline.from_single_file("models/dreamshaperXL_turboDpmppSDE.safetensors", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

        self.width = 1024
        self.height = 1024

        self.steps = 8
        self.guidance_scale = 2
        self.exploit_scale = 0.3

class UltraSpiceTurbo(ModelConfig):
    def __init__(self, seed_prompts:str="seed_prompts/primary.txt"):
        super().__init__(seed_prompts)
        self.pipe = StableDiffusionXLPipeline.from_single_file("models/ultraspiceXLTURBO_v10.safetensors", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

        self.width = 512
        self.height = 512

        self.steps = 6
        self.guidance_scale = 2
        self.exploit_scale = 0.3