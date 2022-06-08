import argparse
import pytorch_lightning as pl
from models.ultra_swin import UltraSwin
from datamodules.EchoNetDataModule import EchoNetDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Train or test")
parser.add_argument("--pretrained", type=str, default=None, help="File pretrained swin")
parser.add_argument("--data_dir", type=str, default="datasets/EchoNet", help="Path ke datasets")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
#parser.add_argument("--embed_dim", type=int, default=128, help="Embed dimension")
parser.add_argument("--frozen_stages", type=int, default=-1 , help="Frozen stages")
parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--accelerator", type=str, default='cpu', help="Accelerator")
parser.add_argument("--dataset_mode", type=str, default='repeat', help="Dataset Mode")
parser.add_argument("--logs_dir", type=str, default='lightning_logs', help="Log dir")
parser.add_argument("--variant", type=str, default='base', help="Variant model")
parser.add_argument("--multi_stage_training", action='store_true', help="Multi stage training")
parser.add_argument("--log", action='store_true', help="log")

params = parser.parse_args()

if __name__ == '__main__':
    mode = params.mode
    data_dir = params.data_dir
    pretrained = params.pretrained
    batch_size = params.batch_size
    #embed_dim = params.embed_dim
    frozen_stages = params.frozen_stages
    ckpt_path = params.ckpt_path
    max_epochs = params.max_epochs
    num_workers = params.num_workers
    accelerator = params.accelerator
    dataset_mode = params.dataset_mode
    logs_dir = params.logs_dir
    multi_stage_training = params.multi_stage_training
    log = params.log
    variant = params.variant

    logger = TensorBoardLogger(save_dir=logs_dir, name="ultraswin")

    data_module = EchoNetDataModule(data_dir=data_dir, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        dataset_mode=dataset_mode)

    if variant == 'small':
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
        embed_dim = 96
    else: # base
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
        embed_dim = 128 

    ultra_swin = UltraSwin(pretrained, 
                    embed_dim=embed_dim, 
                    depths=depths, 
                    num_heads=num_heads, 
                    frozen_stages=frozen_stages, 
                    batch_size=batch_size, 
                    multi_stage_training=multi_stage_training)

    trainer = pl.Trainer(accelerator=accelerator, 
                max_epochs=max_epochs, 
                num_sanity_val_steps=1, 
                auto_scale_batch_size=True, 
                enable_model_summary=True,
                logger=logger,
                precision=16,
                accumulate_grad_batches=2,
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])

    if mode == 'train':
        trainer.fit(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'validate':
        if not log:
            trainer.logger = False

        trainer.validate(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'test':
        if not log:
            trainer.logger = False

        trainer.test(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'predict':
        if not log:
            trainer.logger = False

        predicts = trainer.predict(model=ultra_swin, datamodule=data_module, ckpt_path=ckpt_path)

        for predict in predicts:
            print(predict)
            print('\n')
