import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from downstream.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset,
                                                  BUSIImageDataset,
                                                  AUIDTImageDataset)
from downstream.datasets.data_module import DataModule
from downstream.datasets.transforms import DataTransforms, Moco2Transform
from ViT.vits import create_vit
from downstream.backbone.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--path", type=str,
                        default="/home/sutongkun/VLPv2/VisualGLM/VisualGLM-6B/checkpoints/vit/3.pth")
    parser.add_argument("--test_path", type=str,
                        default='/home/sutongkun/Pretrain_VLP_Project/New_Model/MGCA/data/ckpts/mgca_finetune/36/epoch=49-step=549.ckpt')
    parser.add_argument("--img_encoder", type=str, default='vit')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    # parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--data_pct", type=float, default=1)

    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    elif args.dataset == "busi":
        datamodule = DataModule(BUSIImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    elif args.dataset == "auidt":
        datamodule = DataModule(AUIDTImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.path:
        model, feature_dim = create_vit('base', 224, False, 0, 0)
        state_dict = torch.load(args.path)
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        model, _ = create_vit('base', 224, False, 0, 0)
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)  # 从url中加载vit参数
        state_dict = checkpoint["model"]
        msg = model.load_state_dict(state_dict, strict=False)


    args.model_name = args.img_encoder
    args.backbone = model
    args.in_features = feature_dim
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../checkpoints/classification/{extension}")
    # os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="mgca_finetune",
        save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)
    print(tuner.training_steps)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")
    # trainer.test(tuner, datamodule, ckpt_path=args.test_path)


if __name__ == "__main__":
    cli_main()
