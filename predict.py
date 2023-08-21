# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import zipfile
from cog import BasePredictor, Input, File, Path
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import os
import tempfile
import numpy as np



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
        as_npy: bool = Input(description="Save the the embeddings to a numpy array.", default=False)
    )-> Path: 
        raw_image = Image.open(image)
        inputs = self.__processor(raw_image, return_tensors="pt").to(self.__device)
        image_embeddings = self.__model.get_image_embeddings(inputs["pixel_values"])

        temp_dir =  Path(tempfile.mkdtemp())


        extension = "npy" if as_npy else "pth" 
        filename = f"tensor.{extension}"

        tensor_file = os.path.join(temp_dir, filename)
        output_path = os.path.join(temp_dir, "tensor.zip")

        if as_npy:
            np.save(tensor_file, image_embeddings.detach().cpu().numpy())
        else:
            torch.save(image_embeddings, tensor_file)

        # Compress the tensor file
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_f:
            zip_f.write(tensor_file, arcname=filename)

        return Path(output_path)

