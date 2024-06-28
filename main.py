from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
import uuid
import datetime
import os

def augment(prompt, negative_prompt, image_path, n=1):
    # image = load_image(image_url)

    image = load_image(image_path)
    file_name = image_path.split("/")[-1]

    controlnet_conditioning_scale = 0.5
    controlnet = ControlNetModel.from_pretrained(
        "TheMistoAI/MistoLine",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"output/{timestamp}", exist_ok=True)

    for _ in range(n):
        string_uuid = str(uuid.uuid4())
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        images = pipe(
            prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images
        images[0].save(f"output/{timestamp}/{string_uuid}_{file_name}")


# image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
# prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

image_path = "input/1.jpg"
prompt = "warehouse with boxes and people, different lighting conditions, high quality, professional photography, high resolution, high definition"


# image_path = "input/2.png"
# prompt = "sunflower in a field with bees, different lighting conditions, high quality, professional photography, high resolution, high definition"

# image_path = "input/4.png"
# prompt = "forest scene with lion, different lighting conditions, high quality, professional photography, high resolution, high definition"

negative_prompt = 'low quality, bad quality, sketches, drawings, low resolution, low definition, bad lighting, bad photography'

if __name__ == "__main__":
    augment(prompt, negative_prompt, image_path, 3)