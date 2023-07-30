# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import zipfile
from cog import BasePredictor, Input, File, Path
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import gzip
import os
import pickle
import tempfile



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = SamModel.from_pretrained("./model/", local_files_only=True).to(self.__device)
        self.__processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def predict(
        self,
        image: Path = Input(description="input image"),
    )-> Path: 
        raw_image = Image.open(image)
        inputs = self.__processor(raw_image, return_tensors="pt").to(self.__device)
        image_embeddings = self.__model.get_image_embeddings(inputs["pixel_values"])

        temp_dir =  Path(tempfile.mkdtemp())

        tensor_file = temp_dir / "tensor.pth"
        output_path = temp_dir / "tensor.zip"

        torch.save(image_embeddings, tensor_file)

        # Compress the tensor file
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_f:
            zip_f.write(tensor_file, arcname="tensor.pth")

        return Path(output_path)

