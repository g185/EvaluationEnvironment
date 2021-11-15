from torch.nn.modules.loss import MSELoss
from transformers import BartForConditionalGeneration, AutoConfig, AutoTokenizer
from torch.optim import Adam, AdamW
import pytorch_lightning as pl
import torch
import torch.nn as nn
from latent import IndependentLatentModel
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score

class bartextraggo_module(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = bartextraggo_model()
        self.loss_ma = 0
        self.criterion = MSELoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        ids = batch['input_ids']
        am = batch["attention_mask"]
        keywords = batch["keywords"]
        pdf1 = self.model(source = ids, source_padding_mask = am)
        loss = self.calculate_loss(pdf1, keywords)
        self.loss_ma = 0.99 * self.loss_ma + 0.01 * loss

        y_pred = (pdf1 > 0.5).cpu()
        y_true = (keywords == 1).cpu()
        f1_micro = 0
        f1_macro = 0
        acc = 0
        for i, pred in enumerate(y_pred):
            f1_micro += f1_score(y_true[i], pred, average="macro", zero_division=0)
            f1_macro += f1_score(y_true[i], pred, average="micro", zero_division=0)
            acc += accuracy_score(y_true[i], pred)
        f1_micro = f1_micro/len(y_pred)
        f1_macro = f1_macro/len(y_pred)
        acc = acc/len(y_pred)
        self.log("average_loss", self.loss_ma, prog_bar=True)
        self.log("train_macro_f1", f1_macro, prog_bar=True)
        self.log("train_micro_f1", f1_micro, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_idx: int):
        ids = batch['input_ids']
        am = batch["attention_mask"]
        keywords = batch["keywords"]
        pdf1 = self.forward(source = ids, source_padding_mask = am)
        loss = self.calculate_loss(pdf1, keywords)
        y_pred = (pdf1 > 0.5).cpu()
        y_true = (keywords == 1).cpu()
        f1_micro = 0
        f1_macro = 0
        acc = 0
        for i, pred in enumerate(y_pred):
            f1_micro += f1_score(y_true[i], pred, average="macro", zero_division=0)
            f1_macro += f1_score(y_true[i], pred, average="micro", zero_division=0)
            acc += accuracy_score(y_true[i], pred)
        f1_micro = f1_micro/len(y_pred)
        f1_macro = f1_macro/len(y_pred)
        acc = acc/len(y_pred)
        return loss 
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)


        
    def calculate_loss(self, pdf_1, keywords):
        info_loss = self.criterion(keywords.squeeze(-1), pdf_1.squeeze(-1))
        return info_loss


class bartextraggo_model(nn.Module):

    def __init__(self):
        super().__init__()

        self.config = AutoConfig.from_pretrained('facebook/bart-base', dropout=0.1)
        self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base',
                                                                    config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)

        for par in self.bart_model.model.shared.parameters():
            par.requires_grad = False

        self.latent_model = IndependentLatentModel()

        latent_params = {
                 "selection": 1, 
                 "lasso": 0.002,
                 "lagrange_alpha": 0.5,
                 "lagrange_lr": 0.05,
                 "lambda_init": 0.0015,
                 "lambda_min": 1e-12,
                 "lambda_max": 5.0
        }
        self.selection = latent_params["selection"]
        self.lasso = latent_params["lasso"]
        self.alpha = latent_params["lagrange_alpha"]
        self.lagrange_lr = latent_params["lagrange_lr"]
        self.lambda_init = latent_params["lambda_init"]
        self.lambda_min = latent_params["lambda_min"]
        self.lambda_max = latent_params["lambda_max"]
        
        self.register_buffer('lambda0', torch.full((1,), self.lambda_init))
        self.register_buffer('lambda1', torch.full((1,), self.lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average

    def forward(self, source: torch.Tensor, source_padding_mask: torch.Tensor):
        encoder_outputs = self.bart_model.model.encoder(
            input_ids=source,
            attention_mask=source_padding_mask,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state
        output = self.latent_model(encoder_outputs)
        
        return output

