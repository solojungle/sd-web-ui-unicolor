import kornia
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from module import Decoder, Encoder
from quantize import VectorQuantizer
from vqperceptual import VQLPIPSWithDiscriminator


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 loss_config,
                 n_embed,
                 embed_dim,
                 learning_rate,
                 lr_decay=[10, 1.0],
                 ):
        super().__init__()
        self.automatic_optimization=False
        if loss_config:
            self.loss = VQLPIPSWithDiscriminator(**loss_config)
        self.learning_rate = learning_rate
        self.lr_decay_step, self.lr_decay = lr_decay

        # Color encoder
        self.encoder = Encoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)

        # Decoder
        ddconfig["z_channels"] *= 2
        self.post_quant_conv = torch.nn.Conv2d(2 * embed_dim, ddconfig["z_channels"], 1)
        self.decoder = Decoder(**ddconfig)

        # Gray encoder
        ddconfig['in_channels'] = 1
        ddconfig['out_ch'] = 1
        ddconfig['z_channels'] = embed_dim
        self.gray_encoder = Encoder(**ddconfig)
    

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.gray_encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        ae_lr_scheduler = torch.optim.lr_scheduler.StepLR( opt_ae, step_size=self.lr_decay_step, gamma=self.lr_decay)
        disc_lr_scheduler = torch.optim.lr_scheduler.StepLR( opt_disc, step_size=self.lr_decay_step, gamma=self.lr_decay)
        return [opt_ae, opt_disc], [ae_lr_scheduler, disc_lr_scheduler]
        

    def forward(self, input):
        (quant, diff, _), f_gray = self.encode(input)
        dec = self.decode(quant, f_gray)
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch

        xrec, qloss = self(x)

        # Step LR schedulers
        opt_ae, opt_disc = self.optimizers()
        ae_sch, disc_sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            ae_sch.step()
            disc_sch.step()

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/ae_steps", ae_sch._step_count)
        self.log("train/disc_steps", disc_sch._step_count)
        self.log("train/ae_lr", opt_ae.param_groups[0]['lr'])
        self.log("train/disc_lr", opt_disc.param_groups[0]['lr'])
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        if log_dict_disc != None:
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        

    def validation_step(self, batch, batch_idx):
        x = batch

        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    

    def encode(self, x):
        x_gray = kornia.color.rgb_to_grayscale(x)
        f_gray = self.gray_encoder(x_gray)
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return (quant, emb_loss, info), f_gray

    def decode(self, quant, gray):
        feat = torch.cat([quant, gray], dim=1)
        feat = self.post_quant_conv(feat)
        dec = self.decoder(feat)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
