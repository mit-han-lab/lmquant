import os
import random

import datasets
import yaml

_DESCRIPTION = "Customized local yaml dataset."


class YAMLConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, url: str = None, repeat: int = 1, **kwargs):
        super(YAMLConfig, self).__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.url = url
        self.repeat = repeat
        self.max_dataset_size = max_dataset_size


class DCI(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = YAMLConfig
    BUILDER_CONFIGS = [
        YAMLConfig(name="default", version=VERSION, description="customized yaml dataset"),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "prompt": datasets.Value("string"),
                "meta_path": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(description=_DESCRIPTION, features=features)

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        meta_url = self.config.url
        assert meta_url is not None
        if os.path.exists(meta_url):
            meta_path = meta_url
        else:
            meta_path = dl_manager.download(meta_url)

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"meta_path": meta_path})]

    def _generate_examples(self, meta_path: str):
        meta = yaml.safe_load(open(meta_path, "r"))
        names = list(meta.keys())
        if self.config.max_dataset_size > 0:
            random.Random(0).shuffle(names)
            names = names[: self.config.max_dataset_size]
            names = sorted(names)

        idx = 0
        for i, name in enumerate(names):
            prompt = meta[name]
            for j in range(self.config.repeat):
                yield idx, {"filename": f"{name}-{j}", "prompt": prompt, "meta_path": meta_path}
                idx += 1
