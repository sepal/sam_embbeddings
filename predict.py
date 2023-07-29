# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import json

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
    )-> list: 
        raw_image = Image.open(image)
        inputs = self.__processor(raw_image, return_tensors="pt").to(self.__device)
        image_embeddings = self.__model.get_image_embeddings(inputs["pixel_values"])

        return image_embeddings.cpu().numpy().tolist()

