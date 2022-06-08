import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets.EchoSet import EchoSet

class EchoNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "datasets/EchoNet", 
            batch_size: int = 32, 
            num_workers: int = 8, 
            dataset_mode: str = 'repeat'):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_mode = dataset_mode

        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        #MNIST(self.data_dir, train=True, download=True)
        #MNIST(self.data_dir, train=False, download=True)
        print('prepare data')

    def setup(self, stage = None):
        print(f'setup: {self.data_dir}')

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = EchoSet(root=self.data_dir,
                                split="train",
                                pad=8,
                                augmented=True,
                                #max_data=1000,
                                random_clip=False,
                                dataset_mode=self.dataset_mode)
            
            self.val_set   = EchoSet(root=self.data_dir,
                                split="val",
                                pad=8,
                                random_clip=False,
                                dataset_mode=self.dataset_mode)

        if stage == "validate" or stage is None:
            self.val_set   = EchoSet(root=self.data_dir,
                                split="val",
                                pad=8,
                                #max_data=500,
                                random_clip=False,
                                dataset_mode=self.dataset_mode)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set   = EchoSet(root=self.data_dir,
                                split="test",
                                pad=8,
                                #max_data=100,
                                random_clip=False,
                                dataset_mode=self.dataset_mode)

        if stage == "predict" or stage is None:
            self.predict_set   = EchoSet(root=self.data_dir,
                                split="test",
                                pad=8,
                                #max_data=100,
                                random_clip=False,
                                dataset_mode=self.dataset_mode)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers)