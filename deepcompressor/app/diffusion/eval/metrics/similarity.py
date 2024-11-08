import os

import datasets
import torch
import torchmetrics
import torchvision
from PIL import Image
from torch.utils import data
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm


class MultiImageDataset(data.Dataset):
    def __init__(self, gen_dirpath: str, ref_dirpath_or_dataset: str | datasets.Dataset):
        super(data.Dataset, self).__init__()
        self.gen_names = sorted(
            [name for name in os.listdir(gen_dirpath) if name.endswith(".png") or name.endswith(".jpg")]
        )
        self.gen_dirpath, self.ref_dirpath_or_dataset = gen_dirpath, ref_dirpath_or_dataset
        if isinstance(ref_dirpath_or_dataset, str):
            self.ref_names = sorted(
                [name for name in os.listdir(ref_dirpath_or_dataset) if name.endswith(".png") or name.endswith(".jpg")]
            )
            assert len(self.ref_names) == len(self.gen_names)
        else:
            assert isinstance(ref_dirpath_or_dataset, datasets.Dataset)
            self.ref_names = self.gen_names
            assert len(ref_dirpath_or_dataset) == len(self.gen_names)
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, idx: int):
        if isinstance(self.ref_dirpath_or_dataset, str):
            name = self.ref_names[idx]
            assert name == self.gen_names[idx]
            ref_image = Image.open(os.path.join(self.ref_dirpath_or_dataset, name)).convert("RGB")
        else:
            row = self.ref_dirpath_or_dataset[idx]
            ref_image = row["image"].convert("RGB")
            name = row["filename"] + ".png"
        gen_image = Image.open(os.path.join(self.gen_dirpath, name)).convert("RGB")
        gen_size = gen_image.size
        ref_size = ref_image.size
        if ref_size != gen_size:
            ref_image = ref_image.resize(gen_size, Image.Resampling.BICUBIC)
        gen_tensor = self.transform(gen_image)
        ref_tensor = self.transform(ref_image)
        return [gen_tensor, ref_tensor]


def compute_image_similarity_metrics(
    ref_dirpath_or_dataset: str | datasets.Dataset,
    gen_dirpath: str,
    metrics: tuple[str, ...] = ("psnr", "lpips", "ssim"),
    batch_size: int = 64,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    if len(metrics) == 0:
        return {}
    metric_names = metrics
    metrics: dict[str, torchmetrics.Metric] = {}
    for metric_name in metric_names:
        if metric_name == "psnr":
            metric = PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)).to(device)
        elif metric_name == "lpips":
            metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        elif metric_name == "ssim":
            metric = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
        else:
            raise NotImplementedError(f"Metric {metric_name} is not implemented")
        metrics[metric_name] = metric
    dataset = MultiImageDataset(gen_dirpath, ref_dirpath_or_dataset)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    with torch.no_grad():
        desc = (
            ref_dirpath_or_dataset.config_name
            if isinstance(ref_dirpath_or_dataset, datasets.Dataset)
            else os.path.basename(ref_dirpath_or_dataset)
        ) + " similarity metrics"
        for i, batch in enumerate(tqdm(dataloader, desc=desc)):
            batch = [tensor.to(device) for tensor in batch]
            for metric in metrics.values():
                metric.update(batch[0], batch[1])
    result = {metric_name: metric.compute().item() for metric_name, metric in metrics.items()}
    return result
