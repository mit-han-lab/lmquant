import os
import random

import datasets
import yaml
from PIL import Image

_CITATION = """\
@InProceedings{Urbanek_2024_CVPR,
    author    = {Urbanek, Jack and Bordes, Florian and Astolfi, Pietro and Williamson, Mary and Sharma, Vasu and Romero-Soriano, Adriana},
    title     = {A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {26700-26709}
}
"""

_DESCRIPTION = """\
The Densely Captioned Images dataset, or DCI, consists of 7805 images from SA-1B, 
each with a complete description aiming to capture the full visual detail of what is present in the image. 
Much of the description is directly aligned to submasks of the image.
"""

_HOMEPAGE = "https://github.com/facebookresearch/DCI"

_LICENSE = "Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/DCI/blob/main/LICENSE)"

IMAGE_URL = "https://scontent.xx.fbcdn.net/m1/v/t6/An_zz_Te0EtVC_cHtUwnyNKODapWXuNNPeBgZn_3XY8yDFzwHrNb-zwN9mYCbAeWUKQooCI9mVbwvzZDZzDUlscRjYxLKsw.tar?ccb=10-5&oh=00_AYBnKR-fSIir-E49Q7-qO2tjmY0BGJhCciHS__B5QyiBAg&oe=673FFA8A&_nc_sid=0fdd51"

PROMPT_URLS = {"sDCI": "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.yaml"}


class DCIConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, **kwargs):
        super(DCIConfig, self).__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt


class DCI(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = DCIConfig
    BUILDER_CONFIGS = [DCIConfig(name="sDCI_full", version=VERSION, description="sDCI full prompt set")]
    DEFAULT_CONFIG_NAME = "sDCI"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "image": datasets.Image(),
                "prompt": datasets.Value("string"),
                "meta_path": datasets.Value("string"),
                "image_root": datasets.Value("string"),
                "image_path": datasets.Value("string"),
                "split": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        image_url = IMAGE_URL
        meta_url = PROMPT_URLS[self.config.name]

        meta_path = dl_manager.download(meta_url)
        image_root = dl_manager.download_and_extract(image_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"meta_path": meta_path, "image_root": image_root}
            )
        ]

    def _generate_examples(self, meta_path: str, image_root: str):
        meta = yaml.safe_load(open(meta_path, "r"))
        names = list(meta.keys())
        if self.config.max_dataset_size > 0:
            random.Random(0).shuffle(names)
            names = names[: self.config.max_dataset_size]
            names = sorted(names)

        for i, name in enumerate(names):
            prompt = meta[name]
            image_path = os.path.join(image_root, f"{name}.jpg")
            yield i, {
                "filename": name,
                "image": Image.open(image_path) if self.config.return_gt else None,
                "prompt": prompt,
                "meta_path": meta_path,
                "image_root": image_root,
                "image_path": image_path,
                "split": self.config.name,
            }
