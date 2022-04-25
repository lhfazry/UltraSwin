import argparse
import pytorch_lightning as pl
from models.ultra_swin import UltraSwin
from datamodules.EchoNetDataModule import EchoNetDataModule
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Train or test")
parser.add_argument("--pretrained", type=str, default="pretrained/swin_base_patch4_window12_384_22k.pth", help="File pretrained swin")
parser.add_argument("--data_dir", type=str, default="datasets/EchoNet", help="Path ke datasets")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--embed_dim", type=int, default=96, help="Embed dimension")
parser.add_argument("--frozen_stages", type=int, default="3", help="Frozen stages")
parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--accelerator", type=str, default='cpu', help="Accelerator")
parser.add_argument("--dataset_mode", type=str, default='repeat', help="Dataset Mode")
parser.add_argument("--logs_dir", type=str, default='lightning_logs', help="Log dir")

params = parser.parse_args()

if __name__ == '__main__':
    mode = params.mode
    data_dir = params.data_dir
    pretrained = params.pretrained
    batch_size = params.batch_size
    embed_dim = params.embed_dim
    frozen_stages = params.frozen_stages
    ckpt_path = params.ckpt_path
    max_epochs = params.max_epochs
    num_workers = params.num_workers
    accelerator = params.accelerator
    dataset_mode = params.dataset_mode
    logs_dir = params.logs_dir

    logger = TensorBoardLogger(save_dir=logs_dir, name="ultraswin")

    data_module = EchoNetDataModule(data_dir=data_dir, batch_size=batch_size, 
        num_workers=num_workers, dataset_mode=dataset_mode)
    ultra_swin = UltraSwin(pretrained, embed_dim=embed_dim, depths=[2, 2, 18, 2], 
        frozen_stages=frozen_stages, batch_size=batch_size)

    trainer = pl.Trainer(accelerator=accelerator, max_epochs=max_epochs, 
            num_sanity_val_steps=1, auto_scale_batch_size=True, logger=logger)

    if mode == 'train':
        trainer.fit(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'validate':
        trainer.validate(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'test':
        trainer.test(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'predict':
        predicts = trainer.predict(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)
        print(predicts)
