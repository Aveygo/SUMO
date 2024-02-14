from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

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

class StableCascade(ModelConfig):
    def __init__(self, seed_prompts:str="seed_prompts/primary.txt"):
        super().__init__(seed_prompts)
        self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to("cuda")
        self.decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16).to("cuda")


        self.width = 1024
        self.height = 1024
        self.prior_steps = 10
        self.decoder_steps = 8
        self.guidance_scale = 2
        self.exploit_scale = 0.2

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
        self.exploit_scale = 0.2

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

if __name__ == "__main__":
    import time
    model = StableCascade()

    torch.manual_seed(47)
    latents = torch.randn((1, 4, 256, 256)).cuda().half()

    torch.manual_seed(42)
    a = time.time()
    
    prior_output = model.prior(
        prompt="borris johnson in a bathtub of beans",
        height=1024,
        width=1024,
        #negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=5
    )

    print(prior_output.image_embeddings.shape)
    b = time.time()

    decoder_output = model.decoder(
        image_embeddings=prior_output.image_embeddings.half(),
        prompt="", # Not needed?
        #negative_prompt=negative_prompt,
        latents=latents,
        guidance_scale=1, 
        output_type="pil",
        num_inference_steps=2
    ).images

    print(f"Prior took {b-a:.2f} seconds")
    print(f"Decoder took {time.time() - b:.2f} seconds")

    decoder_output[0].save("imgs/cascade_test.png")