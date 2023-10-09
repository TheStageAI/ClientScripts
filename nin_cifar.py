import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model
import os
import lightning.pytorch as pl
import torch.nn.functional as F


def nin_cifar10(pretrained=True):
    net = get_model("nin_cifar10", pretrained=pretrained)
    net.features.stage2.dropout2 = torch.nn.Identity()
    net.features.stage3.dropout3 = torch.nn.Identity()

    return net


# DATA
batch_size = 128

augmentation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

root = os.path.expanduser("~") + "/datasets/"
train_dataset = torchvision.datasets.CIFAR10(
    root=root, train=True, download=True, transform=augmentation
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

val_dataset = torchvision.datasets.CIFAR10(
    root=root, train=False, download=True, transform=preprocess
)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
loaders = {"train": train_dataloader, "valid": val_dataloader}

# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
class LightModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nin_cifar10()
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(
            f"{prefix}_loss",
            loss_val,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        accuracy = (output.argmax(dim=1) == target).float().mean()
        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def configure_schedulers(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )
        return scheduler


model = LightModel()

# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
trainer = pl.Trainer(limit_train_batches=100, max_epochs=40)
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# ------------------------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------------------------
trainer.test(model, dataloaders=val_dataloader)
