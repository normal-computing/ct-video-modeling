import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

import imageio
import cv2

import numpy as np
import random
import json
import os


TRAIN_TEST_SPLIT_RATIO = 0.8


class UCF101Dataset(Dataset):
    resize_transform = transforms.Compose([transforms.Resize((240, 320))])

    def __init__(
        self,
        path,
        annotation_path,
        metadata=None,
        split="train",
        num_frames=20,
        skip_frames=4,
        fold=1,
    ):
        super().__init__()

        assert split in ["train", "validation"]
        # if metadata is not None:
        #     with open(metadata, "rb") as f:
        #         metadata = pickle.load(f)

        self.skip_frames = skip_frames
        self.num_frames = num_frames

        name = "train" if split == "train" else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(path, *x.split("/")) for x in data]
            selected_files.update(data)
        self.selected_files = list(selected_files)

    def __len__(self):
        return len(self.selected_files)

    def __getitem__(self, idx):
        frames = np.stack(imageio.mimread(self.selected_files[idx], memtest=False))
        frames = frames[:: self.skip_frames]
        if frames.shape[0] < self.num_frames:
            extension = np.tile(
                frames[-1], (self.num_frames - frames.shape[0], 1, 1, 1)
            )
            frames = np.concatenate([frames, extension], axis=0)
        frames = torch.from_numpy(frames[: self.num_frames])
        resized_frames = []
        for frame in frames:
            resized_frames.append(self.resize_transform(frame.moveaxis(-1, 0)))
        frames = torch.stack(resized_frames) / 255.0 * 2.0 - 1.0
        return frames


class MNISTDataset(Dataset):
    def __init__(self, path, split="train", num_frames=20):
        super().__init__()

        assert split in ["train", "validation"]

        dataset = np.load(path).swapaxes(0, 1)
        dataset_size = dataset.shape[0]
        frames_per_vid = dataset.shape[1]
        frame_size = dataset.shape[-2:]
        assert (
            frames_per_vid % num_frames == 0
        ), f"`num_frames` must be divide {frames_per_vid} evenly"
        dataset = dataset.reshape(
            dataset_size * int(frames_per_vid / num_frames), num_frames, *frame_size
        )

        split_size = int(dataset.shape[0] * TRAIN_TEST_SPLIT_RATIO)
        if split == "train":
            self.dataset = dataset[:split_size]
        else:
            self.dataset = dataset[split_size:]

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        frames = torch.from_numpy(self.dataset[idx])
        frames = frames / 255.0 * 2.0 - 1.0
        return frames.unsqueeze(1)


class DAVISDataset(Dataset):
    resize_transforms = {
        "mask_480p": transforms.Compose(
            [
                transforms.Resize(
                    (256, 384), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        ),
        "mask_Full-Resolution": transforms.Compose(
            [
                transforms.Resize(
                    (320, 576), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        ),
        "480p": transforms.Compose([transforms.Resize((256, 384))]),
        "Full-Resolution": transforms.Compose([transforms.Resize((320, 576))]),
    }

    def __init__(
        self,
        path,
        split="train",
        resolution="480p",
        resize=True,
        num_frames=20,
        random=True,
    ):
        super().__init__()
        assert split in ["train", "validation"]
        assert resolution in ["480p", "Full-Resolution"]

        if split == "validation":
            split = "val"

        self.base_path = path
        self.num_frames = num_frames
        self.resolution = resolution
        self.random = random
        self.resize = resize
        if resolution == "480p":
            annotations_file = os.path.join(
                path, "ImageSets", "480p", "2017", f"{split}.txt"
            )
        else:
            annotations_file = os.path.join(
                path, "ImageSets", "Full-Resolution", "2017", f"{split}.txt"
            )
        with open(annotations_file, "r") as f:
            self.sequences = f.read().split("\n")[:-1]

    def __len__(self):
        return len(self.sequences)

    def get_video_from_folder(self, folder, random_subset_start, masks=False):
        files = os.listdir(folder)
        files = sorted(files)[
            random_subset_start : random_subset_start + self.num_frames
        ]
        video = [imageio.imread(os.path.join(folder, file)) for file in files]
        video = torch.from_numpy(np.stack(video)).moveaxis(-1, 1)
        if self.resize:
            resize_transform = self.resize_transforms[self.resolution]
            if masks:
                resize_transform = self.resize_transforms["mask_" + self.resolution]
            video = resize_transform(video)
        if masks:
            video = video[:, :1].repeat((1, 3, 1, 1))
            video[video != 0.0] = 1.0
            return video * 1.0
        return video / 255.0 * 2.0 - 1.0

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        path_to_sequence = os.path.join(
            self.base_path, "Annotations", self.resolution, sequence
        )
        files = os.listdir(path_to_sequence)
        random_subset_start = 10
        if self.random:
            random_subset_start = random.randint(
                0, max(0, len(files) - self.num_frames)
            )
        annotation_video = self.get_video_from_folder(
            path_to_sequence, random_subset_start, masks=True
        )
        path_to_sequence = os.path.join(
            self.base_path, "JPEGImages", self.resolution, sequence
        )
        input_video = self.get_video_from_folder(path_to_sequence, random_subset_start)

        return input_video, annotation_video


class VimeoDataset(Dataset):
    def __init__(self, path, septuplet=False, split="train", apply_transforms=True):
        self.apply_transforms = apply_transforms
        self.split = split
        self.data_root = path
        self.image_root = os.path.join(self.data_root, "sequences")
        if septuplet:
            train_fn = os.path.join(self.data_root, "sep_trainlist.txt")
            test_fn = os.path.join(self.data_root, "sep_testlist.txt")
        else:
            train_fn = os.path.join(self.data_root, "tri_trainlist.txt")
            test_fn = os.path.join(self.data_root, "tri_testlist.txt")
        with open(train_fn, "r") as f:
            self.trainlist = f.read().rstrip().splitlines()
        with open(test_fn, "r") as f:
            self.testlist = f.read().rstrip().splitlines()
        self.load_data()

        self.is_septuplet = septuplet

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.split == "train":
            self.meta_data = self.trainlist[:cnt]
        elif self.split == "test":
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def crop(self, images, h, w):
        ih, iw, _ = images[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        return [img[x : x + h, y : y + w, :] for img in images]

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        if self.is_septuplet:
            imgpaths = [
                os.path.join(imgpath, "im1.png"),
                os.path.join(imgpath, "im2.png"),
                os.path.join(imgpath, "im3.png"),
                os.path.join(imgpath, "im4.png"),
                os.path.join(imgpath, "im5.png"),
                os.path.join(imgpath, "im6.png"),
                os.path.join(imgpath, "im7.png"),
            ]
        else:
            imgpaths = [
                os.path.join(imgpath, "im1.png"),
                os.path.join(imgpath, "im2.png"),
                os.path.join(imgpath, "im3.png"),
            ]
        images = [cv2.imread(p) for p in imgpaths]
        return images

    def __getitem__(self, index):
        images = self.getimg(index)
        if self.split == "train" and self.apply_transforms:
            images = self.crop(images, 240, 320)
            if random.uniform(0, 1) < 0.5:
                images = [image[:, :, ::-1] for image in images]
            if random.uniform(0, 1) < 0.5:
                images = [image[::-1] for image in images]
            if random.uniform(0, 1) < 0.5:
                images = [image[:, ::-1] for image in images]
            if random.uniform(0, 1) < 0.5:
                images = images[::-1]
            # random rotation
            p = random.uniform(0, 1)
            # if p < 0.25:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            #     gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            if p < 0.5:
                images = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
            # elif p < 0.75:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        images = [torch.from_numpy(img.copy()).permute(2, 0, 1) for img in images]
        return torch.stack(images) / 255.0 * 2.0 - 1.0


class YouTubeVOSDataset(Dataset):
    resize_transforms = {
        "masks": transforms.Compose(
            [
                transforms.Resize(
                    (256, 384), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        ),
        "rgb": transforms.Compose([transforms.Resize((256, 384))]),
    }

    def __init__(self, path, split="train", resize=True, num_frames=20):
        assert split in ["train", "test"]
        folder_name = "train" if split == "train" else "valid"

        self.base_path = os.path.join(path, folder_name)
        with open(os.path.join(self.base_path, "meta.json")) as f:
            self.metadata = json.load(f)["videos"]

        self.num_frames = num_frames
        self.resize = resize

    def __len__(self):
        return len(self.metadata.keys())

    def get_video_from_folder(self, folder, files, shape, masks=False):
        video = []
        for file in files:
            if masks:
                file = file.replace(".jpg", ".png")
            fp = os.path.join(folder, file)
            if os.path.exists(fp):
                video.append(imageio.imread(fp))
            else:
                video.append(np.zeros(shape, dtype=np.uint8))
        video = torch.from_numpy(np.stack(video)).moveaxis(-1, 1)
        if self.resize:
            resize_transform = self.resize_transforms["rgb"]
            if masks:
                resize_transform = self.resize_transforms["masks"]
            video = resize_transform(video)
        if masks:
            video = video[:, :1].repeat((1, 3, 1, 1))
            video[video != 0.0] = 1.0
            return video * 1.0
        return video / 255.0 * 2.0 - 1.0

    def __getitem__(self, idx):
        video_id = list(self.metadata.keys())[idx]
        path_to_sequence = os.path.join(self.base_path, "JPEGImages", video_id)
        files = os.listdir(path_to_sequence)
        random_subset_start = random.randint(0, max(0, len(files) - self.num_frames))
        files = sorted(files)[
            random_subset_start : random_subset_start + self.num_frames
        ]

        if len(files) < self.num_frames:
            return self.__getitem__(random.randint(0, len(self) - 1))

        shape = imageio.imread(os.path.join(path_to_sequence, files[0])).shape
        input_video = self.get_video_from_folder(path_to_sequence, files, shape)
        path_to_sequence = os.path.join(self.base_path, "Annotations", video_id)
        annotation_video = self.get_video_from_folder(
            path_to_sequence, files, shape, masks=True
        )

        return input_video, annotation_video


class X4KDataset(Dataset):
    crop_transform = transforms.Compose(
        [transforms.RandomCrop((240, 320), pad_if_needed=True)]
    )

    def __init__(self, path, split="train", num_frames=17):
        assert split in ["train"]

        path = os.path.join(path, f"encoded_{split}")

        vid_collection = []
        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                vid_collection.append(os.path.join(path, folder, file))
        self.vid_collection = vid_collection
        self.num_frames = num_frames

    def __len__(self):
        return len(self.vid_collection)

    def __getitem__(self, idx):
        frames = imageio.mimread(self.vid_collection[idx])
        random_start = random.randint(0, max(0, len(frames) - self.num_frames))
        frames = self.crop_transform(
            torch.from_numpy(
                np.stack(frames[random_start : random_start + self.num_frames])
            ).moveaxis(-1, 1)
        )
        return frames / 255.0 * 2.0 - 1.0


class KTHDataset(Dataset):
    def __init__(self, path, split="train", num_frames=20):
        assert split in {"train", "test"}

        self.split = split
        self.base_path = path
        self.num_frames = num_frames
        self.folder_list = os.listdir(os.path.join(path, split))

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        video = np.load(os.path.join(self.base_path, self.split, self.folder_list[idx]))
        video = torch.from_numpy(video[: self.num_frames]).moveaxis(-1, 1)
        return video / 255.0 * 2.0 - 1.0
