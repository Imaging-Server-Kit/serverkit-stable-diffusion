"""
Algorithm server definition.
Documentation: https://github.com/Imaging-Server-Kit/cookiecutter-serverkit
"""
from typing import List, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
import uvicorn
import skimage.io
import imaging_server_kit as serverkit

import torch
from diffusers import StableDiffusionPipeline

class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    prompt: str = Field(
        ...,
        title="Prompt",
        description="Text prompt",
        json_schema_extra={"widget_type": "str"},
    )

class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str="stable-diffusion",
        parameters_model: Type[BaseModel]=Parameters
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        prompt: str,
        **kwargs
    ) -> List[tuple]:
        """Runs the algorithm."""

        model_id = "sd-legacy/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        generated_image = pipe(prompt).images[0]

        generated_image = np.asarray(generated_image)
        
        return [(generated_image, {}, "image")]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Loads one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images

server = Server()
app = server.app

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)