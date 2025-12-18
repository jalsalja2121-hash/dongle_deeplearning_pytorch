import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class RockPaperScissorsDataModule:
    """
    가위바위보 이미지 데이터셋을 로드하고 전처리하는 모듈

    Args:
        data_root (str): 데이터셋의 루트 디렉토리 (기본값: '.')
        batch_size (int): 배치 크기 (기본값: 32)
        image_size (tuple): 이미지 리사이즈 크기 (기본값: (224, 224))
        num_workers (int): DataLoader 워커 수 (기본값: 0)
    """

    def __init__(self, data_root='.', batch_size=32, image_size=(224, 224), num_workers=0):
        self.data_root = data_root
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using Pytorch version: {torch.__version__}, Device: {self.device}")

        # 데이터셋 및 DataLoader 초기화
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.train_dataset_loader = None
        self.validation_dataset_loader = None
        self.test_dataset_loader = None
        self.labels_map = None

    def _get_train_transforms(self):
        """학습용 데이터 증강 및 전처리 transform"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def _get_validation_transforms(self):
        """검증/테스트용 전처리 transform (증강 없음)"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def setup(self):
        """데이터셋 및 DataLoader 초기화"""
        train_config = self._get_train_transforms()
        validation_config = self._get_validation_transforms()

        # ImageFolder를 이용하면 이미지들이 각 Class에 해당하는 폴더에 각각 나누어져 있을 때
        # 이것을 dataset 형태로 쉽게 불러올 수 있음
        train_path = os.path.join(self.data_root, 'train')
        valid_path = os.path.join(self.data_root, 'valid')
        test_path = os.path.join(self.data_root, 'test')

        self.train_dataset = datasets.ImageFolder(train_path, train_config)
        self.validation_dataset = datasets.ImageFolder(valid_path, validation_config)
        self.test_dataset = datasets.ImageFolder(test_path, validation_config)

        # DataLoader 설정
        self.train_dataset_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.validation_dataset_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_dataset_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        # ImageFolder의 속성 값인 class_to_idx를 할당
        self.labels_map = {v: k for k, v in self.train_dataset.class_to_idx.items()}

        self._print_dataset_info()

    def _print_dataset_info(self):
        """데이터셋 정보 출력"""
        # train data 개수
        train_path = os.path.join(self.data_root, 'train')
        train_paper_list = os.listdir(os.path.join(train_path, 'paper'))
        train_rock_list = os.listdir(os.path.join(train_path, 'rock'))
        train_scissor_list = os.listdir(os.path.join(train_path, 'scissors'))

        # validation data 개수
        valid_path = os.path.join(self.data_root, 'valid')
        val_paper_list = os.listdir(os.path.join(valid_path, 'paper'))
        val_rock_list = os.listdir(os.path.join(valid_path, 'rock'))
        val_scissor_list = os.listdir(os.path.join(valid_path, 'scissors'))

        print(len(train_paper_list), len(train_rock_list), len(train_scissor_list))
        print(len(val_paper_list), len(val_rock_list), len(val_scissor_list))

    def visualize_batch(self, num_images=16, dataset='train', figsize=(6, 7)):
        """
        배치에서 샘플 이미지 시각화

        Args:
            num_images (int): 표시할 이미지 개수 (기본값: 16)
            dataset (str): 'train', 'validation', 'test' 중 선택 (기본값: 'train')
            figsize (tuple): Figure 크기 (기본값: (6, 7))
        """
        if self.train_dataset_loader is None:
            raise RuntimeError("먼저 setup() 메서드를 호출하여 데이터를 로드하세요.")

        # 데이터셋 선택
        loader_map = {
            'train': self.train_dataset_loader,
            'validation': self.validation_dataset_loader,
            'test': self.test_dataset_loader
        }
        loader = loader_map.get(dataset, self.train_dataset_loader)

        # 1개의 배치를 추출
        images, labels = next(iter(loader))

        figure = plt.figure(figsize=figsize)
        cols, rows = 4, 4

        # 이미지 출력
        for i in range(1, min(num_images, len(images), cols*rows) + 1):
            sample_idx = torch.randint(len(images), size=(1,)).item()
            img, label = images[sample_idx], labels[sample_idx].item()

            figure.add_subplot(rows, cols, i)
            plt.title(self.labels_map[label])
            plt.axis("off")

            # 본래 이미지의 shape은 (3, 224, 224) 인데,
            # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (224, 224, 3)으로 shape 변경을 한 후 시각화
            plt.imshow(torch.permute(img, (1, 2, 0)))

        plt.show()

    def get_train_loader(self):
        """학습용 DataLoader 반환"""
        return self.train_dataset_loader

    def get_validation_loader(self):
        """검증용 DataLoader 반환"""
        return self.validation_dataset_loader

    def get_test_loader(self):
        """테스트용 DataLoader 반환"""
        return self.test_dataset_loader

    def get_class_names(self):
        """클래스 이름 목록 반환"""
        return list(self.labels_map.values())

    def get_num_classes(self):
        """클래스 개수 반환"""
        return len(self.labels_map)
