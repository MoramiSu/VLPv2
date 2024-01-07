import datetime
import os
from argparse import ArgumentParser

import segmentation_models_pytorch as smp
import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from datasets.data_module import DataModule
from datasets.segmentation_dataset import (RSNASegmentDataset,
                                                SIIMImageDataset,
                                                BUSISegmentDataset,
                                                DDTISegmentDataset
                                           )
from datasets.segmentation_dataset import seg_collate_fn
from models.backbones.transformer_seg import SETRModel
from models.mgca.mgca_module import MGCA
from models.ssl_segmenter import SSLSegmenter

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser(
        "Finetuning of semantic segmentation task for MGCA")
    parser.add_argument("--base_model", type=str,
                        default="resnet50", help="resnet50 or vit")
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/sutongkun/VLPv2/GLoRIA/data/ckpt/gloria_pretrain_1.0/2023_10_16_21_46_29/epoch=3-step=427.ckpt")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_pct", type=float, default=1)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "siim":
        datamodule = DataModule(SIIMImageDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNASegmentDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
    elif args.dataset == "busi":
        datamodule = DataModule(BUSISegmentDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
    elif args.dataset == "ddti":
        datamodule = DataModule(DDTISegmentDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)

    # mgca = MGCA.load_from_checkpoint(args.ckpt_path)
    mgca = MGCA()
    encoder = mgca.img_encoder_q.model
    # encoder.load_state_dict(torch.load('/home/sutongkun/VLPv2/MRM/output/vit.pth'))
    # encoder.load_state_dict(torch.load(args.ckpt_path), strict=False)

    if args.base_model == "vit":
        args.seg_model = SETRModel(
            patch_size=(16, 16),
            in_channels=3,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64]
        )
        args.seg_model.encoder_2d.bert_model = encoder  # 将MGCA的vit作为分割模型的vit

        for param in args.seg_model.encoder_2d.bert_model.parameters():
            param.requires_grad = False

    elif args.base_model == "resnet50":
        # FIXME: fix this later
        args.seg_model = smp.Unet(
            args.base_model, encoder_weights=None, activation=None)

        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path)
            ckpt_dict = dict()
            for k, v in ckpt["state_dict"].items():
                if k.startswith("gloria.img_encoder.model"):
                    new_k = ".".join(k.split(".")[3:])
                    new_k = new_k.replace("blocks", "layer")
                    ckpt_dict[new_k] = v

            ckpt_dict["fc.bias"] = None
            ckpt_dict["fc.weight"] = None

            args.seg_model.encoder.load_state_dict(ckpt_dict)
            # Freeze encoder
            for param in args.seg_model.encoder.parameters():
                param.requires_grad = False

    model = SSLSegmenter(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/segmentation/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == "__main__":
    cli_main()
