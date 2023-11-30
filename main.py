from vidgen.utils import (
    parse_devices,
    load_config,
    save_config,
    instantiate_from_config,
    setup_log_dir,
)

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import torch

import numpy as np
import datetime
import argparse
import glob
import os

# os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("-d", "--debug", default=False, action="store_true")
args = parser.parse_args()

if args.resume is None:
    assert (
        args.base is not None
    ), "Configs not specified, specify at least resume or base"
    config = load_config(args.base)
else:
    assert os.path.exists(
        args.resume
    ), "Provided path to resume training does not exist"
    config_paths = glob.glob(os.path.join(args.resume, "*.yaml"))
    assert len(config_paths) == 1, "Too many possible configs to resume from"
    config = load_config(config_paths[0])

torch.set_float32_matmul_precision("medium")
torch.manual_seed(int(config["seed"]))
np.random.seed(int(config["seed"]))

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
experiment_name = config.get("experiment_name", None)
log_dir_name = setup_log_dir(
    config.get("log_dir_name", "logs"),
    timestamp,
    experiment_name=experiment_name,
    debug=args.debug,
    resume=args.resume,
)

# Save training config
if args.resume is None:
    save_config(config, f"{log_dir_name}/{os.path.basename(args.base)}")

if __name__ == "__main__":
    model = instantiate_from_config(config["model"])
    dataset = instantiate_from_config(config["data"])

    logger = CSVLogger(name="csvlogger", save_dir=log_dir_name, version="")
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        ModelCheckpoint(
            dirpath=f"{log_dir_name}/checkpoints/trainstep_checkpoints",
            filename="{epoch:06}-{step:09}",
            every_n_train_steps=config["batch_frequency"],
            save_last=True,
            verbose=True,
            save_weights_only=True,
        ),
        ModelCheckpoint(
            dirpath=f"{log_dir_name}/checkpoints",
            filename="{epoch:06}",
            verbose=True,
            save_last=True,
            save_on_train_epoch_end=True,
            save_weights_only=False,
        ),
    ]
    logger_config = config.get("logger", None)
    if logger_config:
        logger_config["params"]["log_directory"] = log_dir_name
        samples_logger = instantiate_from_config(logger_config)
        callbacks.append(samples_logger)

    trainer_kwargs = {
        "max_epochs": config["max_epochs"],
        "log_every_n_steps": config["batch_frequency"],
        "gradient_clip_val": config["gradient_clip_val"],
        "accumulate_grad_batches": config["accumulate_grad_batches"],
        "accelerator": "gpu",
        "devices": args.devices,
        "strategy": "ddp_find_unused_parameters_false",
        "num_sanity_val_steps": 0,
    }

    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks
    trainer = pl.Trainer(**trainer_kwargs)

    try:
        resume_ckpt = None
        if args.resume is not None:
            resume_ckpt = os.path.join(args.resume, "checkpoints", "last.ckpt")
        trainer.fit(model, dataset, ckpt_path=resume_ckpt)
    finally:
        if trainer.global_rank == 0:
            final_ckpt = os.path.join(log_dir_name, "checkpoints", "last.ckpt")
            trainer.save_checkpoint(final_ckpt)
