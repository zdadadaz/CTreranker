import torch
import pytorch_lightning as pl
from transformers import (BertForNextSentencePrediction, get_linear_schedule_with_warmup)


class CrossEncoder(torch.nn.Module):
    def __init__(self,
                 encoder_name_or_dir,
                 encoder_config=None,
                 cache_dir=None):
        super().__init__()
        self.encoder = BertForNextSentencePrediction.from_pretrained(encoder_name_or_dir,
                                                                     config=encoder_config,
                                                                     cache_dir=cache_dir)

    def forward(self, inputs, labels=None):
        outputs = self.encoder(**inputs, labels=labels)
        return outputs


class BertReranker(pl.LightningModule):
    def __init__(self,
                 encoder_name_or_dir,
                 encoder_config=None,
                 cache_dir=None,
                 lr=2e-5,
                 warm_up_steps_percent=10,
                 num_gpus=1,
                 batch_size=64,
                 num_epochs=2,
                 train_set_size=532761,  # ms marco train size
                 num_neg_per_pos=4
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = CrossEncoder(encoder_name_or_dir,
                                    encoder_config,
                                    cache_dir)

    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        total_steps = self.hparams.num_epochs * \
                      int(self.hparams.train_set_size / (self.hparams.batch_size * self.hparams.num_gpus))

        if self.hparams.warm_up_steps_percent == 0:
            return optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(float(self.hparams.warm_up_steps_percent)/100*total_steps), num_training_steps=total_steps
        )
        schedulers = [{
            'scheduler': scheduler,
            'name': 'warm_up_lr',
            'interval': 'step'
        }]
        optimizers = [optimizer]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.encoder(inputs, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)

        return loss

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs

    def get_scores(self, inputs):
        outputs = self.encoder(inputs)
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1)[:, 1]

        return scores
