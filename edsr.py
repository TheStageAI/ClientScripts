import argparse
import torch
from super_image import EdsrModel, ImageLoader
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from super_image import Trainer, TrainingArguments
from datasets import load_dataset


parser = argparse.ArgumentParser(description="INN EDSR")
parser.add_argument("--results", default="results", help="save checkpoint directory")
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--scale", default=4, type=int, help="super resolution scale (default: 4)"
)
parser.add_argument("-b", "--batch-size", default=32, type=int, metavar="N")
parser.add_argument("-w", "--workers", default=48, type=int)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
args = parser.parse_args()

# DATA
augmented_dataset = load_dataset(
    "eugenesiow/Div2k", f"bicubic_x{args.scale}", split="train"
).map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset)
eval_dataset = EvalDataset(
    load_dataset("eugenesiow/Div2k", f"bicubic_x{args.scale}", split="validation")
)

# MODEL
model = EdsrModel.from_pretrained("eugenesiow/edsr", scale=args.scale)

# TRAIN
training_args = TrainingArguments(
    output_dir=args.results,
    num_train_epochs=args.epochs,
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    dataloader_num_workers=args.workers,
    dataloader_pin_memory=True,
    gamma=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

if not args.evaluate:
    trainer.train()

# EVAL
trainer.eval(1)


# image = Image.open("0853x4.png")
# inputs = ImageLoader.load_image(image).cuda()
# preds = model(inputs)
# ImageLoader.save_image(preds, f"{args.results}/scaled_{args.scale}x.png")
# ImageLoader.save_compare(
#     inputs, preds, f"{args.results}/scaled_{args.scale}x_compare.png"
# )
