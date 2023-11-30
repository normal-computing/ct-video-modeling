from ..img_utils import postprocess

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import imageio
import random
import cv2

import warnings
import os


class VideoPredictionLogger(Callback):
    def __init__(
        self,
        log_directory,
        batch_frequency,
        downsample_factor=1.0,
    ):
        super().__init__()
        assert 0.0 < downsample_factor <= 1.0, "Downsample factor must be (0.0, 1.0]"
        self.downsample_factor = downsample_factor

        self.make_logging_dir(log_directory)
        self.batch_frequency = batch_frequency

    @rank_zero_only
    def make_logging_dir(self, log_directory):
        self.log_dir = f"{log_directory}/videos"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    @rank_zero_only
    def make_log(self, pl_module, batch, epoch_num, step, split="train"):
        batch = batch[: min(3, batch.size(0))]
        log_dict = pl_module.log_video_sample(batch, None)

        if "reconstructions" not in log_dict:
            warnings.warn("`reconstructions` key not in log dictionary")

        input_video = self.postprocess_for_save(batch)

        for key, val in log_dict.items():
            video = self.postprocess_for_save(val)

            if key in ["reconstructions", "samples"]:
                video = np.concatenate([input_video, video], axis=-3)

            # Remove channel dimension if input is grayscale
            if video.shape[-1] == 1:
                video = video.squeeze(-1)

            output_path = os.path.join(
                self.log_dir,
                f"epoch_{epoch_num:05}_step_{step:06}_{key}_{split}.gif",
            )
            imageio.mimsave(output_path, video.astype(np.uint8), loop=0)

    def postprocess_for_save(self, video):
        # Postprocess -- move off GPU, inverse normalize
        video = postprocess(video)
        # Resize videos if specified
        if self.downsample_factor != 1.0:
            video = self.resize_batch_frames(video, self.downsample_factor)
        # Restack videos to output
        return np.concatenate([v for v in video], axis=-2)

    def resize_batch_frames(self, batch, factor):
        resized_batch = []
        for video in batch:
            resized_video = []
            for frame in video:
                add_last_channel_back = True if frame.shape[-1] == 1 else False
                resized_frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
                # For grayscale images, resizing automatically removes the last dimension
                if add_last_channel_back:
                    resized_frame = np.expand_dims(resized_frame, axis=-1)
                resized_video.append(resized_frame)
            resized_batch.append(np.stack(resized_video))
        return resized_batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            if hasattr(self, "last_global_step"):
                if self.last_global_step != trainer.global_step:
                    self.make_log(
                        pl_module,
                        batch,
                        trainer.current_epoch,
                        trainer.global_step,
                        split="train",
                    )

        self.last_global_step = trainer.global_step

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            if hasattr(self, "last_global_step"):
                if self.last_global_step != trainer.global_step:
                    self.make_log(
                        pl_module,
                        batch,
                        trainer.current_epoch,
                        trainer.global_step,
                        split="validation",
                    )

        self.last_global_step = trainer.global_step


class VideoInterpolationLogger(Callback):
    def __init__(
        self,
        log_directory,
        batch_frequency,
        downsample_factor=1.0,
    ):
        super().__init__()
        assert 0.0 < downsample_factor <= 1.0, "Downsample factor must be (0.0, 1.0]"
        self.downsample_factor = downsample_factor

        self.make_logging_dir(log_directory)
        self.batch_frequency = batch_frequency

    @rank_zero_only
    def make_logging_dir(self, log_directory):
        self.log_dir = f"{log_directory}/videos"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    @rank_zero_only
    def make_log(self, pl_module, batch, epoch_num, step, split="train"):
        batch = batch[random.choice(list(batch.keys()))]

        batch = batch[: min(3, batch.size(0))]
        log_dict = pl_module.log_video_sample(batch, None)

        # if "reconstructions" not in log_dict:
        #     warnings.warn("`reconstructions` key not in log dictionary")

        input_video = self.postprocess_for_save(batch)

        for key, val in log_dict.items():
            video = self.postprocess_for_save(val)

            if key in ["reconstructions", "samples"]:
                video = np.concatenate([input_video, video], axis=-3)

            # Remove channel dimension if input is grayscale
            if video.shape[-1] == 1:
                video = video.squeeze(-1)

            output_path = os.path.join(
                self.log_dir,
                f"epoch_{epoch_num:05}_step_{step:06}_{key}_{split}.gif",
            )
            imageio.mimsave(output_path, video.astype(np.uint8), loop=0)

    def postprocess_for_save(self, video):
        # Postprocess -- move off GPU, inverse normalize
        video = postprocess(video)
        # Resize videos if specified
        if self.downsample_factor != 1.0:
            video = self.resize_batch_frames(video, self.downsample_factor)
        # Restack videos to output
        return np.concatenate([v for v in video], axis=-2)

    def resize_batch_frames(self, batch, factor):
        resized_batch = []
        for video in batch:
            resized_video = []
            for frame in video:
                add_last_channel_back = True if frame.shape[-1] == 1 else False
                resized_frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
                # For grayscale images, resizing automatically removes the last dimension
                if add_last_channel_back:
                    resized_frame = np.expand_dims(resized_frame, axis=-1)
                resized_video.append(resized_frame)
            resized_batch.append(np.stack(resized_video))
        return resized_batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            if hasattr(self, "last_global_step"):
                if self.last_global_step != trainer.global_step:
                    self.make_log(
                        pl_module,
                        batch,
                        trainer.current_epoch,
                        trainer.global_step,
                        split="train",
                    )

        self.last_global_step = trainer.global_step

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if (trainer.global_step + 1) % self.batch_frequency == 0:
    #         if hasattr(self, "last_global_step"):
    #             if self.last_global_step != trainer.global_step:
    #                 self.make_log(
    #                     pl_module,
    #                     batch,
    #                     trainer.current_epoch,
    #                     trainer.global_step,
    #                     split="validation",
    #                 )

    #     self.last_global_step = trainer.global_step


class MaskPropagationLogger(Callback):
    def __init__(
        self,
        log_directory,
        batch_frequency,
        downsample_factor=1.0,
    ):
        super().__init__()
        assert 0.0 < downsample_factor <= 1.0, "Downsample factor must be (0.0, 1.0]"
        self.downsample_factor = downsample_factor

        self.make_logging_dir(log_directory)
        self.batch_frequency = batch_frequency

    @rank_zero_only
    def make_logging_dir(self, log_directory):
        self.log_dir = f"{log_directory}/videos"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    @rank_zero_only
    def make_log(self, pl_module, batch, epoch_num, step, split="train"):
        batch, *_ = batch
        x, annotations = batch

        x, annotations = x[: min(3, x.size(0))], annotations[: min(3, x.size(0))]
        log_dict = pl_module.log_video_sample(x, annotations)

        input_video = self.postprocess_for_save(x)
        input_masks = self.postprocess_for_save(annotations, masks=True)

        for key, val in log_dict.items():
            video = self.postprocess_for_save(val, masks=True)

            if key == "reconstructions":
                video = np.concatenate([input_masks, video], axis=-3)
            else:
                video = np.concatenate([input_video, input_masks, video], axis=-3)

            # Remove channel dimension if input is grayscale
            if video.shape[-1] == 1:
                video = video.squeeze(-1)

            output_path = os.path.join(
                self.log_dir,
                f"epoch_{epoch_num:05}_step_{step:06}_{key}_{split}.gif",
            )
            imageio.mimsave(output_path, video.astype(np.uint8), loop=0)

    def postprocess(self, frames, masks=False):
        frames = frames.clamp(-1.0, 1.0)
        if masks:
            frames = frames * 255.0
            # frames[frames < 127.5] = 0.0
            # frames[frames != 0.0] = 255.0
        else:
            frames = (frames + 1.0) / 2.0 * 255.0
        return frames.moveaxis(2, -1).detach().cpu().numpy()

    def postprocess_for_save(self, video, masks=False):
        # Postprocess -- move off GPU, inverse normalize
        video = self.postprocess(video, masks=masks)
        # Resize videos if specified
        if self.downsample_factor != 1.0:
            video = self.resize_batch_frames(video, self.downsample_factor)
        # Restack videos to output
        return np.concatenate([v for v in video], axis=-2)

    def resize_batch_frames(self, batch, factor):
        resized_batch = []
        for video in batch:
            resized_video = []
            for frame in video:
                add_last_channel_back = True if frame.shape[-1] == 1 else False
                resized_frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
                # For grayscale images, resizing automatically removes the last dimension
                if add_last_channel_back:
                    resized_frame = np.expand_dims(resized_frame, axis=-1)
                resized_video.append(resized_frame)
            resized_batch.append(np.stack(resized_video))
        return resized_batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            if hasattr(self, "last_global_step"):
                if self.last_global_step != trainer.global_step:
                    self.make_log(
                        pl_module,
                        batch,
                        trainer.current_epoch,
                        trainer.global_step,
                        split="train",
                    )

        self.last_global_step = trainer.global_step

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            if hasattr(self, "last_global_step"):
                if self.last_global_step != trainer.global_step:
                    self.make_log(
                        pl_module,
                        batch,
                        trainer.current_epoch,
                        trainer.global_step,
                        split="validation",
                    )

        self.last_global_step = trainer.global_step
