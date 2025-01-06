from composer import Trainer
from composer.algorithms import StochasticDepth
from composer.loggers import WandBLogger
from composer.optim import DecoupledSGDW
from composer.utils import dist
import torch
from torchvision import transforms
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Convert
from ffcv.pipeline.operation import Operation
from pathlib import Path

# 1. Prepare the dataset for FFCV
DATASET_PATH = Path("/path/to/imagenet")
FFCV_TRAIN_PATH = DATASET_PATH / "train.ffcv"
FFCV_VAL_PATH = DATASET_PATH / "val.ffcv"

BATCH_SIZE = 96
NUM_WORKERS = 4

if not FFCV_TRAIN_PATH.exists() or not FFCV_VAL_PATH.exists():
    train_dataset = ImageNetKaggle(str(DATASET_PATH), "train")
    val_dataset = ImageNetKaggle(str(DATASET_PATH), "val")

    # Writing datasets to FFCV format
    train_writer = DatasetWriter(str(FFCV_TRAIN_PATH), {
        'image': RGBImageField(write_mode='smart', max_resolution=256, compress_probability=0.5),
        'label': IntField()
    })

    val_writer = DatasetWriter(str(FFCV_VAL_PATH), {
        'image': RGBImageField(write_mode='smart', max_resolution=256),
        'label': IntField()
    })

    train_writer.from_indexed_dataset(train_dataset)
    val_writer.from_indexed_dataset(val_dataset)

# 2. Define FFCV loaders
ffcv_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = Loader(
    str(FFCV_TRAIN_PATH),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    order=OrderOption.RANDOM,
    pipelines={
        'image': [
            ToTensor(),
            Convert(torch.float32),
            ToDevice(ffcv_device, non_blocking=True)
        ],
        'label': [
            ToTensor(),
            ToDevice(ffcv_device, non_blocking=True)
        ]
    },
    drop_last=True
)

val_loader = Loader(
    str(FFCV_VAL_PATH),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    order=OrderOption.SEQUENTIAL,
    pipelines={
        'image': [
            ToTensor(),
            Convert(torch.float32),
            ToDevice(ffcv_device, non_blocking=True)
        ],
        'label': [
            ToTensor(),
            ToDevice(ffcv_device, non_blocking=True)
        ]
    },
    drop_last=False
)

# 3. Define the model
model = ResNet50(num_classes=len(train_dataset.syn_to_class))
model = model.to(ffcv_device)

# 4. Configure the optimizer, scheduler, and algorithms
optimizer = DecoupledSGDW(
    params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

stochastic_depth = StochasticDepth(stochastic_method='batch', drop_rate=0.2)

# 5. Integrate MosaicML Trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    optimizers=optimizer,
    max_duration="1ep",
    algorithms=[stochastic_depth],
    device="gpu" if torch.cuda.is_available() else "cpu",
    precision="amp",
    schedulers=[scheduler],
    loggers=[WandBLogger(project="imagenet-training")],
    save_folder="checkpoints",
    save_interval="1ep"
)

# 6. Train the model
trainer.fit()
