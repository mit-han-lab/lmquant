# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""COCO"""
import json
import os
import random
from pathlib import Path

import datasets
from PIL import Image

_CITATION = """
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  eprinttype = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
MS COCO is a large-scale object detection, segmentation, and captioning dataset.
COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.
"""

_HOMEPAGE = "https://cocodataset.org/#home"

_LICENSE = "CC BY 4.0"


_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "validation": "http://images.cocodataset.org/zips/val2014.zip",
}

_KARPATHY_FILES_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

_FEATURES = datasets.Features(
    {
        "filepath": datasets.Value("string"),
        "filename": datasets.Value("string"),
        "image": datasets.Image(),
        "image_path": datasets.Value("string"),
        "image_root": datasets.Value("string"),
        "prompt": datasets.Value("string"),
        "prompt_id": datasets.Value("int32"),
        "imgid": datasets.Value("int32"),
        "split": datasets.Value("string"),
        "cocoid": datasets.Value("int32"),
        "sentences_raw": [datasets.Value("string")],
        "sentids": [datasets.Value("int32")],
        "sentences_sentid": [datasets.Value("int32")],
        "sentences_tokens": [[datasets.Value("string")]],
    }
)


def hash_string_to_int(s: str) -> int:
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


class COCOConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, **kwargs):
        super(COCOConfig, self).__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt


class COCO(datasets.GeneratorBasedBuilder):
    """COCO"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = COCOConfig
    BUILDER_CONFIGS = [
        COCOConfig(name="COCO_val", version=VERSION, description="COCO validation prompt set"),
        COCOConfig(name="COCO_train", version=VERSION, description="COCO train prompt set"),
        COCOConfig(name="COCO_full", version=VERSION, description="COCO full prompt set"),
    ]
    DEFAULT_CONFIG_NAME = "COCO_val"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        annotation_file = os.path.join(dl_manager.download_and_extract(_KARPATHY_FILES_URL), "dataset_coco.json")
        image_folders = {k: Path(v) for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()}

        if self.config.name == "COCO_full":
            split_keys = ["validation", "train"]
        else:
            split_keys = [self.config.name.split("_")[-1]]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "image_folders": image_folders,
                    "split_keys": split_keys,
                },
            ),
        ]

    def _generate_examples(
        self, annotation_file: str, image_folders: dict[str, str], split_keys: list[str] | tuple[str, ...]
    ):
        with open(annotation_file, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)
        metas = []
        for split_key in split_keys:
            for image_metadata in annotations["images"]:
                if split_key == "train":
                    if image_metadata["split"] != "train" and image_metadata["split"] != "restval":
                        continue
                elif split_key == "val":
                    if image_metadata["split"] != "val":
                        continue
                elif split_key == "test":
                    if image_metadata["split"] != "test":
                        continue

                metas.append(image_metadata)

        if self.config.max_dataset_size > 0:
            random.Random(0).shuffle(metas)
            metas = metas[: self.config.max_dataset_size]
            metas = sorted(metas, key=lambda x: x["filename"])

        for i, meta in enumerate(metas):
            if "val2014" in meta["filename"]:
                image_root = os.path.join(image_folders["validation"], "val2014")
            else:
                image_root = os.path.join(image_folders["train"], "train2014")
            filename = meta["filename"].replace(".jpg", "").replace(".png", "")
            image_path = os.path.join(image_root, filename + ".jpg")

            sentences_raw = [caption["raw"] for caption in meta["sentences"]]
            prompt_id = hash_string_to_int(filename) % len(sentences_raw)
            prompt = sentences_raw[prompt_id]

            yield i, {
                "filename": filename,
                "image": Image.open(image_path) if self.config.return_gt else None,
                "image_path": image_path,
                "image_root": image_root,
                "prompt": prompt,
                "prompt_id": prompt_id,
                "imgid": meta["imgid"],
                "split": self.config.name,
                "coco_id": meta["cocoid"],
                "sentences_raw": sentences_raw,
                "sentids": meta["sentids"],
                "sentences_sentid": [caption["sentid"] for caption in meta["sentences"]],
                "sentences_tokens": [caption["tokens"] for caption in meta["sentences"]],
            }
