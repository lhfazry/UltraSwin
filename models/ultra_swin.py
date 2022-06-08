from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from torch.nn import functional as F
from backbones.swin_transformer import SwinTransformer3D
from einops import rearrange
from utils.tensor_utils import Reduce
from losses.mse import mse_loss
from losses.r2 import r2_loss
from sklearn.metrics import r2_score
from losses.rmse import RMSE


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
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False, 
                 batch_size=8, 
                 multi_stage_training=False):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.multi_stage_training = multi_stage_training

        #self.train_rmse = RMSE()
        #self.train_mae = torchmetrics.MeanAbsoluteError()
        #self.train_r2 = torchmetrics.R2Score()

        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_r2 = torchmetrics.R2Score()

        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_r2 = torchmetrics.R2Score()

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
        
        self.extremas = nn.Sequential(
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            nn.LayerNorm(4*embed_dim),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=4*embed_dim, out_features=2*embed_dim, bias=True),
            nn.LayerNorm(2*embed_dim),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=2*embed_dim, out_features=1, bias=True)
        )
        
        self.ejection = nn.Sequential(
            nn.LayerNorm(8*embed_dim),
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(4*embed_dim),
            nn.Linear(in_features=4*embed_dim, out_features=2*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(2*embed_dim),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=2*embed_dim, out_features=1, bias=True),

            Reduce(),
            nn.Sigmoid()
        )

        '''
        self.ejection = nn.Sequential(
            nn.LayerNorm(8*embed_dim),
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(4*embed_dim),
            nn.Linear(in_features=4*embed_dim, out_features=2*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(2*embed_dim),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=2*embed_dim, out_features=1, bias=True),

            Reduce(),
            nn.Sigmoid()
        )
        '''

        '''
        self.ejection2 = nn.Sequential(
            nn.LayerNorm(8*embed_dim),
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            #nn.Dropout(p=0.5),
            
            nn.LayerNorm(4*embed_dim),
            nn.Linear(in_features=4*embed_dim, out_features=16, bias=True),
            #nn.Dropout(p=0.5),

            nn.Linear(in_features=16, out_features=1, bias=True),
            Reduce(),
            #nn.Tanh()
        )
        '''

        #self.dropout = nn.Dropout(p=0.5)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # output size ==> d' x h' x w'
        #self.ef_regressor = nn.Linear(in_features=8*embed_dim, out_features=1, bias=True)
        self.reduce = Reduce()

    def forward_features(self, x):
        #print(f'1: {x.shape}')
        # (N, D, C, H, W)
        x = rearrange(x, 'n d c h w -> n c d h w')

        x = self.swin_transformer(x) # n c d h w ==> torch.Size([1, 768, 32, 4, 4])
        #x = x.mean(2) # n c h w
        #x = x.flatten(-2) # n c hxw
        #print(f'x: {x.shape}')
        #x = x.transpose(1, 2) # n hxw c

        return x

    def forward_head(self, x):
        # input ==> n c d h w
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = x.flatten(-2).mean(3) # n d c
        #x = self.avg_pool(x) # n c 1 1 1
        #x = self.dropout(x)
        #x = x.view(x.shape[0], -1) # n d
        #print(x.shape)
        #class_vec = self.extremas(x) # n d 1
        
        ef = self.ejection(x) # n 1

        return ef 

    def forward_head_old(self, x):
        # input ==> n c d h w
        #x = rearrange(x, 'n c d h w -> n c d h w')

        x = self.avg_pool(x) # n c 1 1 1
        #x = self.dropout(x)
        x = x.view(x.shape[0], -1) # n c

        x = self.ejection(x)

        return x # n c

    def forward(self, x):
        #x1, x2, x3, x4 = self.forward_features(x) # n c d h w

        #if self.multi_stage_training:
        #    x = self.forward_head(x4) # n f
        #else:
        #    x = self.forward_head(x4) # n f

        #print(x.shape)

        x = self.forward_features(x) # n c
        #print(x)
        x = self.forward_head(x) # n 1

        return x

    def training_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch

        #print(f'ejection before: {ejection}')
        #ejection = (2 * ejection.type(torch.float32) / 100.) - 1
        #print(f'ejection after: {ejection}')
        ef_label = ejection.type(torch.float32) / 100.

        #print(f'nlabel: {nlabel.shape}')
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')

        #classes_vec, ef_pred = self(nvideo)
        #print(f'classes_vec: {classes_vec.shape}')
        ef_pred = self(nvideo)
        #print(f'ef_pred size: {ef_pred.shape}')

        #print(f'ejection: {ejection.data}')
        #print(f'y_hat: {y_hat.data}')
        #loss1 = F.cross_entropy(classes_vec.view(-1, 3), nlabel.view(-1))
        #loss2 = F.mse_loss(ef_pred, ef_label)

        #loss = loss1 + loss2
        #loss = F.huber_loss(y_hat, ejection)
        loss = F.mse_loss(ef_pred, ef_label)
        
        #self.train_mse(y_hat, ejection)
        #self.train_mae(y_hat, ejection)
        #r2loss = r2_loss(y_hat, ejection)

        self.log('loss', loss, on_epoch=True, batch_size=self.batch_size)
        #self.log('train_mse', self.train_mse, on_step=True, on_epoch=False, batch_size=self.batch_size)
        #self.log('train_mae', self.train_mse, on_step=True, on_epoch=False, batch_size=self.batch_size)
        #self.log('train_r2', r2loss, on_step=True, on_epoch=False, batch_size=self.batch_size)
        
        #tensorboard_logs = {'loss':{'train': loss.detach() } }
        #return {"loss": loss, 'log': tensorboard_logs }
        return loss

    def validation_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        #ejection = (2 * ejection.type(torch.float32) / 100.) - 1
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo)
        loss = F.mse_loss(ef_pred, ef_label)

        self.val_rmse(ef_pred, ef_label)
        self.val_mae(ef_pred, ef_label)
        self.val_r2(ef_pred, ef_label)
        #r2loss = r2_score(y_hat, ejection)

        self.log('val_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_rmse', self.val_rmse, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_mae', self.val_mae, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_r2', self.val_r2, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        #tensorboard_logs = {'loss':{'val': loss.detach() } }
        #return {"val_loss": loss, 'log': tensorboard_logs }
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        #ejection = (2 * ejection.type(torch.float32) / 100.) - 1
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo) 
        loss = F.mse_loss(ef_pred, ef_label)
        
        self.test_rmse(ef_pred, ef_label)
        self.test_mse(ef_pred, ef_label)
        self.test_mae(ef_pred, ef_label)
        self.test_r2(ef_pred, ef_label)
        #r2loss = r2_score(y_hat, ejection)

        self.log('test_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_rmse', self.test_rmse, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_mse', self.test_mse, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_mae', self.test_mae, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_r2', self.test_r2, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo) 

        loss = F.mse_loss(ef_pred, ejection)
        return {'filename': filename, 'EF': ef_label * 100., 'Pred': ef_pred * 100., 'loss': loss * 100.}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)

        return [optimizer], [lr_scheduler]