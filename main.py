"""

This was more of an experiment rather than a full-fledged application. Only the bare necessities are here.

The user guided position, velocity, and momentum is stored in saved_latents/primary.pickle, which may be loaded
by uncommenting the line: profile.load("saved_latents/primary.pickle")

Make sure to play around with models.ModelConfig.exploit_scale, especially optimizer.AdamOptimizer.lr for best results.
Increasing either will make results appear faster at the cost of a sub-optimal solution. 

"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import io
from fastapi.middleware.cors import CORSMiddleware

from latent.latent import Latent
from optimizer import AdamOptimizer
from algo import QueuedImageGen

from models import StabilityTurbo, RealVisTurbo, DreamShaperTurbo, UltraSpiceTurbo
from latent.SDXLTurboInput import SDXLTurboInput
profile = AdamOptimizer(DreamShaperTurbo(), SDXLTurboInput, lr=0.05)

#from models import StableCascade
#from latent.StableCascadeInput import StableCascadeInput
#profile = AdamOptimizer(StableCascade(), StableCascadeInput, lr=5e-3)

#profile.load("saved_latents/primary.pickle")

algo = QueuedImageGen(profile)

app = FastAPI()

origins = [
    "http://192.168.1.65:8080",
    "http://localhost:8080",
    "http://0.0.0.0:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def show(latent:Latent):
    imgio = io.BytesIO()
    latent.generate().save(imgio, 'JPEG')
    imgio.seek(0)
    return StreamingResponse(content=imgio, media_type="image/jpeg")

@app.get("/pull/{image_id}.jpeg")
async def pull_image(image_id:int):
    global algo
    global profile
    profile.save()
    return show(algo.pull(image_id))

@app.get("/push/{image_id}.jpeg")
async def push_image(image_id:int):
    global algo
    if algo.push(image_id):
        return Response(status_code=204)
    
    return Response(status_code=409)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)