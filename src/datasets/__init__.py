"""ImageFolder 기반 이미지 분류 데이터셋 모듈."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageClassificationDataModule:
    """train/valid(or val)/test 폴더를 사용하는 공통 데이터 모듈."""

    def __init__(
        self,
        data_root,
        batch_size=32,
        image_size=(224, 224),
        num_workers=0,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        augment=True,
        **_,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.image_size = tuple(image_size)
        self.num_workers = num_workers
        self.augment = augment
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def _transform(self, training=False):
        steps = [transforms.Resize(self.image_size)]
        if training and self.augment:
            steps.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
            ])
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std),
        ])
        return transforms.Compose(steps)

    def _split_path(self, *names):
        for name in names:
            candidate = self.data_root / name
            if candidate.is_dir():
                return candidate
        expected = ", ".join(str(self.data_root / name) for name in names)
        raise FileNotFoundError(f"Dataset split not found. Expected one of: {expected}")

    def setup(self):
        train_path = self._split_path("train")
        validation_path = self._split_path("valid", "val")
        test_path = self._split_path("test")

        self.train_dataset = datasets.ImageFolder(
            train_path, transform=self._transform(training=True)
        )
        self.validation_dataset = datasets.ImageFolder(
            validation_path, transform=self._transform()
        )
        self.test_dataset = datasets.ImageFolder(
            test_path, transform=self._transform()
        )

        if self.train_dataset.classes != self.validation_dataset.classes:
            raise ValueError("Train and validation class folders do not match.")
        if self.train_dataset.classes != self.test_dataset.classes:
            raise ValueError("Train and test class folders do not match.")

        print(f"Classes: {self.train_dataset.classes}")
        print(
            "Samples: "
            f"train={len(self.train_dataset)}, "
            f"validation={len(self.validation_dataset)}, "
            f"test={len(self.test_dataset)}"
        )

    def _loader(self, dataset, shuffle=False):
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a DataLoader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_train_loader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def get_validation_loader(self):
        return self._loader(self.validation_dataset)

    def get_test_loader(self):
        return self._loader(self.test_dataset)

    def get_num_classes(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the class count.")
        return len(self.train_dataset.classes)


DATASET_REGISTRY = {
    "image_folder": ImageClassificationDataModule,
    "rps": ImageClassificationDataModule,
    "chest_xray": ImageClassificationDataModule,
}


def create_dataset(config):
    """설정의 type 값에 맞는 데이터 모듈을 생성한다."""
    dataset_type = config.get("type", "image_folder")
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available datasets: {list(DATASET_REGISTRY)}"
        )
    parameters = {key: value for key, value in config.items() if key != "type"}
    return DATASET_REGISTRY[dataset_type](**parameters)


__all__ = [
    "ImageClassificationDataModule",
    "DATASET_REGISTRY",
    "create_dataset",
]
