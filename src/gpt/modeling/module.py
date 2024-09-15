import lightning as L
import torch

from gpt.modeling.model import GPT
from gpt.modeling.optim import GPTLearningRateScheduler
from gpt.modeling.schemas import GPTConfig, OptimizerConfig


class GPTModule(L.LightningModule):
    def __init__(self, config: GPTConfig, optimizer_config: OptimizerConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = GPT(config)
        # self.model = torch.compile(self.model)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.trainer.is_global_zero:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )

        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizer_config.min_lr,
            betas=(self.optimizer_config.beta_1, self.optimizer_config.beta_2),
            eps=self.optimizer_config.eps,
            fused=False,
        )

        scheduler = GPTLearningRateScheduler(
            optimizer=optimizer,
            max_lr=self.optimizer_config.max_lr,
            min_lr=self.optimizer_config.min_lr,
            warmup_steps=self.optimizer_config.warmup_steps,
            max_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
