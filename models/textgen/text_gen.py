import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything

from datasets.data_module import DataModule
from datasets.caption_dataset import CaptionDataset, caption_collate_fn
from datasets.transforms import DataTransforms
from models.mgca.mgca_module import MGCA
from models.textgen.captioner import Captioner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cli_main():
    parser = ArgumentParser(
        "Finetuning of image captioning task for MGCA")
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/sutongkun/VLPv2/MGCA/data/ckpts/MGCA/2023_10_19_17_09_47/epoch=24-step=4149.ckpt")
    parser.add_argument("--dataset", type=str, default="ultra")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_pct", type=float, default=1)
    parser.add_argument("--prompt", type=str, default='生成中文超声报告：')
    parser.add_argument("--beam_size", type=int, default=5)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    datamodule = DataModule(CaptionDataset, caption_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    model = MGCA.load_from_checkpoint(args.ckpt_path, strict=True)

    model = Captioner(model, **args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/caption/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    trainer = Trainer.from_argparse_args(args=args)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
