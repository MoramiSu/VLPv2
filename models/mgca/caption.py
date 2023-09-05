import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from MGCA.datasets.data_module import DataModule
from MGCA.datasets.caption_dataset import UltrasonicCaptioningDataset, multimodal_collate_fn
from MGCA.datasets.transforms import DataTransforms
from MGCA.models.mgca.mgca_module import MGCA
from MGCA.models.captioner import Captioner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config(object):
    def __init__(self, attention_probs_dropout_prob, hidden_act, hidden_drop_out, hidden_size, initializer_Range, intermediate_size, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, num_decoder_layers, max_target_embeddings, max_words):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_drop_out
        self.hidden_size = hidden_size
        self.initializer_range = initializer_Range
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.max_target_embeddings = max_target_embeddings
        self.max_words = max_words

def cli_main():
    parser = ArgumentParser(
        "Finetuning of image captioning task for MGCA")
    parser.add_argument("--base_model", type=str,
                        default="resnet50", help="resnet50 or vit")
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/sutongkun/Pretrain_VLP_Project/MGCA_Ultrasonic/MGCA/data/ckpts/MGCA/1_1/last.ckpt")
    parser.add_argument("--dataset", type=str, default="ultra")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_pct", type=float, default=1)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "ultra":
        datamodule = DataModule(UltrasonicCaptioningDataset, multimodal_collate_fn,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)

    mgca = MGCA.load_from_checkpoint(args.ckpt_path)
    # mgca = MGCA()
    encoder = mgca.img_encoder_q
    word_embedding_weight = mgca.text_encoder_q.model.embeddings.word_embeddings.weight
    positional_embedding_weight = mgca.text_encoder_q.model.embeddings.position_embeddings.weight

    config = Config(attention_probs_dropout_prob=0.1, hidden_act='gelu', hidden_drop_out=0.1, hidden_size=768, initializer_Range=0.02, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12, type_vocab_size=2, vocab_size=21128, num_decoder_layers=1, max_target_embeddings=512, max_words=200)

    model = Captioner(encoder, word_embedding_weight, positional_embedding_weight, config)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/caption/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="BLEU_4", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=5),
        EarlyStopping(monitor="BLEU_4", min_delta=0.,
                      patience=10, verbose=False, mode="max")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="segmentation", save_dir=logger_dir,
        name=f"MGCA_{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == "__main__":
    cli_main()
