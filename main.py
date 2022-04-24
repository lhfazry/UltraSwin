import argparse
import pytorch_lightning as pl
from models.ultra_swin import UltraSwin
from datamodules.EchoNetDataModule import EchoNetDataModule

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Train or test")
parser.add_argument("--pretrained", type=str, default="pretrained/swin_base_patch4_window12_384_22k.pth", help="File pretrained swin")
parser.add_argument("--data_dir", type=str, default="datasets/EchoNet", help="Path ke datasets")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--embed_dim", type=int, default=96, help="Embed dimension")
parser.add_argument("--frozen_stages", type=int, default="3", help="Frozen stages")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Frozen stages")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--accelerator", type=str, default='cpu', help="Accelerator")

params = parser.parse_args()

if __name__ == '__main__':
    mode = params.mode
    data_dir = params.data_dir
    pretrained = params.pretrained
    batch_size = params.batch_size
    embed_dim = params.embed_dim
    frozen_stages = params.frozen_stages
    checkpoint_path = params.checkpoint_path
    max_epochs = params.max_epochs
    num_workers = params.num_workers
    accelerator = params.accelerator

    data_module = EchoNetDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    ultra_swin = UltraSwin(pretrained, embed_dim=embed_dim, depths=[2, 2, 18, 2], 
        frozen_stages=frozen_stages)

    trainer = pl.Trainer(accelerator=accelerator, max_epochs=max_epochs, 
            num_sanity_val_steps=1, auto_scale_batch_size=True)

    if mode == 'train':
        trainer.fit(model=ultra_swin, datamodule=data_module, ckpt_path=checkpoint_path)
        trainer.test()

    if mode == 'test':
        trainer.test(model=ultra_swin, datamodule=data_module, ckpt_path=checkpoint_path)
