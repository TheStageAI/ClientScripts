import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from catalyst import dl
from pytorchcv.model_provider import get_model
import os


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
model = nin_cifar10()

# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
cross_entropy = nn.CrossEntropyLoss()
log_dir = "./logs/cifar"
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)
callbacks = [
    dl.AccuracyCallback(
        input_key="logits", target_key="targets", topk=(1,), num_classes=10
    ),
    dl.SchedulerCallback(mode="batch", loader_key="train", metric_key="loss"),
]
loggers = []
epochs = 30

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
epoch_len = len(train_dataloader)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [epoch_len * 2, epoch_len * 5, epoch_len * 6, epoch_len * 8], gamma=0.33
)
runner.train(
    model=model,
    criterion=cross_entropy,
    optimizer=opt,
    scheduler=sched,
    loaders=loaders,
    num_epochs=epochs,
    callbacks=callbacks,
    loggers=loggers,
    logdir=log_dir,
    valid_loader="valid",
    valid_metric="loss",
    verbose=True,
)

# ------------------------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------------------------
metrics = runner.evaluate_loader(
    model=model, loader=loaders["valid"], callbacks=callbacks[:1]
)
