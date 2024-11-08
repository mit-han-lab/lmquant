import os

import datasets
import numpy as np
import torch
import torchmetrics
import torchvision
from PIL import Image
from torch.utils import data
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore
from tqdm import tqdm


class PromptImageDataset(data.Dataset):
    def __init__(self, ref_dataset: datasets.Dataset, gen_dirpath: str):
        super(data.Dataset, self).__init__()
        self.ref_dataset, self.gen_dirpath = ref_dataset, gen_dirpath
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.ref_dataset)

    def __getitem__(self, idx: int):
        row = self.ref_dataset[idx]
        gen_image = Image.open(os.path.join(self.gen_dirpath, row["filename"] + ".png")).convert("RGB")
        gen_tensor = torch.from_numpy(np.array(gen_image)).permute(2, 0, 1)
        prompt = row["prompt"]
        return [gen_tensor, prompt]


def compute_image_multimodal_metrics(
    ref_dataset: datasets.Dataset,
    gen_dirpath: str,
    metrics: tuple[str, ...] = ("clip_iqa", "clip_score"),
    batch_size: int = 64,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    if len(metrics) == 0:
        return {}
    metric_names = metrics
    metrics: dict[str, torchmetrics.Metric] = {}
    for metric_name in metric_names:
        if metric_name == "clip_iqa":
            metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        elif metric_name == "clip_score":
            metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        else:
            raise NotImplementedError(f"Metric {metric_name} is not implemented")
        metrics[metric_name] = metric
    dataset = PromptImageDataset(ref_dataset, gen_dirpath)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"{ref_dataset.config_name} multimodal metrics")):
            batch[0] = batch[0].to(device)
            for metric_name, metric in metrics.items():
                if metric_name == "clip_iqa":
                    metric.update(batch[0].to(torch.float32))
                else:
                    prompts = list(batch[1])
                    metric.update(batch[0], prompts)
    result = {metric_name: metric.compute().mean().item() for metric_name, metric in metrics.items()}
    return result
