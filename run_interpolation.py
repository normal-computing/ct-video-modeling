from vidgen.data.loaders import VimeoDataset
from vidgen.utils import instantiate_from_config, load_config
from vidgen.utils.img_utils import postprocess

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from einops import rearrange

import torch
import numpy as np
from tqdm import tqdm
import math
import fire

import copy
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


datasets = {
    "vimeo": VimeoDataset(
        "dataset/vimeo_triplet", split="test", apply_transforms=False
    ),
}
ssim = SSIM(data_range=1.0)


def model_setup(config_path, ckpt_path):
    config = load_config(config_path)
    checkpoint = torch.load(ckpt_path)

    model = instantiate_from_config(config["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model


def postprocess(frames):
    return (frames.cpu().clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0


def average_results(metrics):
    metrics = copy.deepcopy(metrics)
    for key in metrics.keys():
        if len(metrics[key]["psnr"]):
            metrics[key]["psnr"] = np.mean(metrics[key]["psnr"])
        else:
            metrics[key]["psnr"] = 0.0

        if len(metrics[key]["ssim"]):
            metrics[key]["ssim"] = np.mean(metrics[key]["ssim"])
        else:
            metrics[key]["ssim"] = 0.0

    return metrics


def grab_mid_frames(frames, n):
    return torch.cat(
        [frames[:, i : i + (n - 1)] for i in range(1, frames.size(1), n)], 1
    )


def main(ckpt_path, config_path, dataset_name="vimeo"):
    model = model_setup(config_path, ckpt_path)
    dataset = datasets[dataset_name]

    print("Model successfully loaded!")

    metrics = {2: {"ssim": [], "psnr": []}}

    pbar = tqdm(range(len(dataset)))
    for idx in pbar:
        original_frames = dataset.__getitem__(idx)
        if isinstance(original_frames, tuple):
            original_frames, _ = original_frames
        x = original_frames.to(device).unsqueeze(0)

        for key in metrics.keys():
            assert x.size()

            output_frames = model(x[:, ::key], key)

            cut_x = grab_mid_frames(cut_x, key)
            output_frames = grab_mid_frames(output_frames, key)

            gt = postprocess("ours", cut_x) / 255.0
            mid = postprocess(output_frames) / 255.0

            metrics[key]["psnr"].append(-10 * math.log10(((gt - mid) ** 2).mean()))
            metrics[key]["ssim"].append(
                ssim(
                    rearrange(mid, "b t c h w -> (b t) c h w"),
                    rearrange(gt, "b t c h w -> (b t) c h w"),
                )
            )

            pbar.set_postfix(
                {
                    f"ssim": metrics[key]["ssim"][-1],
                    f"psnr": metrics[key]["psnr"][-1],
                }
            )

    metrics = average_results(metrics)

    print(
        "PSNR Metrics:",
        {key: value["psnr"] for key, value in metrics.items()},
        "SSIM Metrics:",
        {key: value["ssim"] for key, value in metrics.items()},
    )

    with open(f"evaluation_results/{dataset_name}.txt", "w") as f:
        f.write(json.dumps(str(metrics)))


if __name__ == "__main__":
    fire.Fire(main)
