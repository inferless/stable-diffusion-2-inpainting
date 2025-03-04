import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import PIL
import requests
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

class InferlessPythonModel:
    @staticmethod
    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    def initialize(self):
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        image_url = inputs["image_url"]
        mask_url = inputs["mask_url"]
        init_image = InferlessPythonModel.download_image(image_url).resize((512, 512))
        mask_image = InferlessPythonModel.download_image(mask_url).resize((512, 512))
        inpaint_image = self.pipe(
            prompt=prompt, image=init_image, mask_image=mask_image
        ).images[0]
        buff = BytesIO()
        inpaint_image.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        return {"generated_image_base64" : img_str.decode('utf-8')}

    def finalize(self):
        self.pipe = None
