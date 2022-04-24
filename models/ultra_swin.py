import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from torch.nn import functional as F
from backbones.swin_transformer import SwinTransformer3D
from einops import rearrange
from utils.tensor_utils import Reduce
from losses.mse import mse_loss

class UltraSwin(pl.LightningModule):
    def __init__(self,
                 pretrained=None,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False, 
                 batch_size=8):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.train_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        #self.train_r2 = torchmetrics.R2Score()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        #self.val_r2 = torchmetrics.R2Score()

        self.swin_transformer = SwinTransformer3D(
            pretrained=pretrained,
            pretrained2d=True,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.ejection = nn.Sequential(
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            nn.LayerNorm(4*embed_dim),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=4*embed_dim, out_features=1, bias=True),
            Reduce(),
            nn.Sigmoid()
        )



    def forward_features(self, x):
        #print(f'1: {x.shape}')
        x = rearrange(x, 'n d c h w -> n c d h w')

        #print(f'2: {x.shape}')
        x = self.swin_transformer(x) # n c d h w ==> torch.Size([1, 768, 32, 4, 4])
        x = x.mean(2) # n c h w
        x = x.flatten(-2) # n c hxw
        #print(f'x: {x.shape}')
        x = x.transpose(1, 2) # n hxw c
        #x = rearrange(x, 'n c d hw -> n d hw c')
        #x = self.avgpool(x).squeeze() # n d hw

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.ejection(x)

        return x

    def training_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch

        ejection = (ejection / 100).type(torch.float32)
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')

        y_hat = self(nvideo) 

        #print(f'ejection: {ejection.data}')
        #print(f'y_hat: {y_hat.data}')

        loss = mse_loss(y_hat, ejection)

        self.train_mse(y_hat, ejection)
        self.train_mae(y_hat, ejection)
        #self.train_r2(y_hat, ejection)

        self.log('train_loss', loss, batch_size=self.batch_size)
        self.log('train_mse', self.train_mse, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log('train_mae', self.train_mse, on_step=True, on_epoch=False, batch_size=self.batch_size)
        #self.log('train_r2', self.train_r2, on_step=True, on_epoch=False, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        ejection = (ejection / 100).type(torch.float32)
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')

        y_hat = self(nvideo) 
        loss = mse_loss(y_hat, ejection)
        
        self.val_mse(y_hat, ejection)
        self.val_mae(y_hat, ejection)
        #self.val_r2(y_hat, ejection)

        self.log('val_loss', loss, batch_size=self.batch_size)
        self.log('val_mse', self.val_mse, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log('val_mae', self.val_mae, on_step=True, on_epoch=True, batch_size=self.batch_size)
        #self.log('val_r2', self.val_r2, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=10e-5)