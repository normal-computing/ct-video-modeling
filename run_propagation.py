from vidgen.utils import instantiate_from_config, load_config
from vidgen.data.loaders import DAVISDataset

from segmentation_mask_overlay import overlay_masks_video

import torch.nn.functional as F
import numpy as np
import imageio
import torch

from tqdm import tqdm
import fire
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

os.makedirs("outputs", exist_ok=True)

datasets = {
    "davis": DAVISDataset(
        "dataset/DAVIS",
        "validation",
        resolution="480p",
        num_frames=15,
    ),
}


def model_setup(config_path, ckpt_path):
    config = load_config(config_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = instantiate_from_config(config["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model


def binarize_output(masks):
    masks[masks < 0.3] = 0.0
    masks[masks != 0.0] = 1.0
    return masks


def j_metric(pred, target):
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)

    return torch.sum(intersection) / torch.sum(union)


def f_metric(pred, target):
    true_positive = torch.sum(torch.logical_and(pred, target))
    false_positive = torch.sum(torch.logical_and(torch.logical_not(pred), target))
    false_negative = torch.sum(torch.logical_and(pred, torch.logical_not(target)))

    return true_positive / (true_positive + 0.5 * (false_positive + false_negative))


def main(ckpt_path, config_path, dataset_name="davis"):
    model = model_setup(config_path, ckpt_path)
    dataset = datasets[dataset_name]

    metrics = {"bce": [], "j": [], "f": []}
    pbar = tqdm(range(len(dataset)))
    for idx in pbar:
        frames, masks = dataset[idx]
        frames, masks = frames.unsqueeze(0), masks.unsqueeze(0)
        frames, masks = frames.to(device), masks.to(device)[:, :, :1]

        if len(masks[:, 0].unique()) == 1:
            continue

        ts = torch.arange(0, frames.size(1)).to(frames)
        output = model(frames, masks[:, :1], ts, logits=True)[:, :, :1]
        output = binarize_output(output)

        metrics["bce"].append(
            F.binary_cross_entropy(output[:, 1:], masks[:, 1:]).item()
        )
        metrics["j"].append(j_metric(output[:, 1:], masks[:, 1:]).item())
        metrics["f"].append(f_metric(output[:, 1:], masks[:, 1:]).item())

        pbar.set_postfix(
            {
                "bce": metrics["bce"][-1],
                "f": metrics["f"][-1],
                "j": metrics["j"][-1],
            }
        )

        frames = frames.squeeze(0).moveaxis(1, -1).cpu().numpy()
        output = np.concatenate(
            [masks[0, :1, 0].cpu().numpy(), output[0, 1:, 0].cpu().numpy()], axis=0
        )
        output = np.expand_dims(output, axis=-1)

        output = overlay_masks_video(frames, output, mask_weight=0.8)
        imageio.mimwrite(
            f"outputs/pred_{idx}.mp4", output[:, :, output.shape[2] // 2 :]
        )
        output = overlay_masks_video(
            frames,
            np.expand_dims(masks[0, :, 0].cpu().numpy(), axis=-1),
            mask_weight=0.8,
        )
        imageio.mimwrite(f"outputs/gt_{idx}.mp4", output[:, :, output.shape[2] // 2 :])

    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])

    print(f"Final metrics: {metrics}")

    with open(f"evaluation_results/propagation_{dataset_name}.txt", "w") as f:
        f.write(json.dumps(str(metrics)))


if __name__ == "__main__":
    fire.Fire(main)
