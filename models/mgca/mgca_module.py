import datetime
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from datasets.data_module import DataModule
from datasets.pretrain_dataset import MultimodalPretrainingDataset, multimodal_collate_fn
from datasets.transforms import DataTransforms
from models.backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist
from transformers import GPT2LMHeadModel, GPT2Config, TextGenerationPipeline

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 img_encoder: str = "resnet_50",
                 freeze_bert: bool = False,
                 queue_size = 100,
                 query_num = 32,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()  #保持超参，方便重新实例化

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)  # 图像编码器
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert, split='text_encoder')  # 文本编码器
        self.qformer = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert, split='qformer')
        config = GPT2Config.from_json_file(os.path.join(BASE_DIR, "../../configs/gpt_config.json"))
        self.decoder = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "../backbones/GPT"),
                                                       config=config)

        self.register_buffer("img_queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("text0_queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("text1_queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("qfeat0_queue", torch.randn(self.hparams.queue_size, self.hparams.query_num, self.hparams.emb_dim))
        self.register_buffer("qfeat1_queue", torch.randn(self.hparams.queue_size, self.hparams.query_num, self.hparams.emb_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.img_queue = F.normalize(self.img_queue, dim=1)
        self.text0_queue = F.normalize(self.text0_queue, dim=1)
        self.text1_queue = F.normalize(self.text1_queue, dim=1)
        self.qfeat0_queue = F.normalize(self.qfeat0_queue, dim=2)
        self.qfeat1_queue = F.normalize(self.qfeat1_queue, dim=2)

        # patch local attention layer
        # self.patch_local_atten_layer = nn.MultiheadAttention(
        #     self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        # self.word_local_atten_layer = nn.MultiheadAttention(
        #     self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        # self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)  # 用线性变换将编码特征变换到原型
        # if self._use_ddp_or_dpp2(self.trainer):  # 是否已进行多机分布式计算初始化
            # self.get_assignments = self.distributed_sinkhorn  # 分布式sinkhorn算法
        # else:
            # self.get_assignments = self.sinkhorn  # 单机sinkhorn算法

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        global_img0_feat, local_img0_feat = self.img_encoder_q(batch['img0'])
        global_img1_feat, local_img1_feat = self.img_encoder_q(batch['img0'])
        global_text_feat, local_text_feat = self.text_encoder_q(ids=batch['report'], attn_mask=batch['attn'])

        #### Contrastive Learning ####

        cl_img0_feat = self.img_encoder_q.embed_0(global_img0_feat)
        cl_img0_feat = F.normalize(cl_img0_feat, dim=-1)
        cl_img1_feat = self.img_encoder_q.embed_0(global_img1_feat)
        cl_img1_feat = F.normalize(cl_img1_feat, dim=-1)
        cl_img_feat = (cl_img0_feat + cl_img1_feat) / 2
        cl_text_feat = self.text_encoder_q.embed_0(global_text_feat)
        cl_text_feat = F.normalize(cl_text_feat, dim=-1)

        cl_img_feat_all = torch.cat([cl_img_feat, self.img_queue.clone().detach()], dim=0)
        cl_text_feat_all = torch.cat([cl_text_feat, self.text0_queue.clone().detach()], dim=0)

        bs = cl_text_feat.size(0)
        cl_labels = torch.arange(bs).type_as(cl_text_feat).long()

        scores = cl_img_feat.mm(cl_text_feat_all.t()) / self.hparams.softmax_temperature
        scores_t = cl_text_feat.mm(cl_img_feat_all.t()) / self.hparams.softmax_temperature
        closs0 = F.cross_entropy(scores, cl_labels)
        closs1 = F.cross_entropy(scores_t, cl_labels)
        c_loss = closs0 + closs1

        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, cl_labels, top_k=(1, 5))  # 图像2文本
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores_t, cl_labels, top_k=(1, 5))  # 文本2图像
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        #### Qformer Contrastive Learning ####

        # lm_img0_feat = torch.cat([global_img0_feat.unsqueeze(1), local_img0_feat], dim=1)
        lm_img0_feat = self.img_encoder_q.embed_1(local_img0_feat)
        lm_img0_feat = F.normalize(lm_img0_feat, dim=-1)
        # lm_img1_feat = torch.cat([global_img1_feat.unsqueeze(1), local_img1_feat], dim=1)
        lm_img1_feat = self.img_encoder_q.embed_1(local_img1_feat)
        lm_img1_feat = F.normalize(lm_img1_feat, dim=-1)

        qformer_input = torch.zeros([bs, self.hparams.query_num, self.hparams.hidden_dim]).to(lm_img0_feat.device)
        global_q_feat0, local_q_feat0 = self.qformer(inputs_embeds=qformer_input, encoder_hidden_states=lm_img0_feat)
        q_feat0 = torch.cat([global_q_feat0.unsqueeze(1), local_q_feat0], dim=1)
        global_q_feat1, local_q_feat1 = self.qformer(inputs_embeds=qformer_input, encoder_hidden_states=lm_img1_feat)
        q_feat1 = torch.cat([global_q_feat1.unsqueeze(1), local_q_feat1], dim=1)
        qcl_q_feat0 = self.qformer.embed_0(q_feat0)
        qcl_q_feat0 = F.normalize(qcl_q_feat0, dim=-1)
        qcl_q_feat1 = self.qformer.embed_0(q_feat1)
        qcl_q_feat1 = F.normalize(qcl_q_feat1, dim=-1)

        qcl_text_feat = self.text_encoder_q.embed_1(global_text_feat)
        qcl_text_feat = F.normalize(qcl_text_feat, dim=-1)

        qcl_q_feat0_all = torch.cat([qcl_q_feat0, self.qfeat0_queue.clone().detach()], dim=0)
        qcl_q_feat1_all = torch.cat([qcl_q_feat1, self.qfeat1_queue.clone().detach()], dim=0)
        qcl_text_feat_all = torch.cat([qcl_text_feat, self.text1_queue.clone().detach()], dim=0)

        sim_mat0 = torch.matmul(qcl_text_feat,
                                torch.transpose(qcl_q_feat0_all, 1, 2)) / self.hparams.softmax_temperature
        qscores0 = torch.max(sim_mat0, dim=-1)[0]
        qscores0 = qscores0.t()
        sim_mat1 = torch.matmul(qcl_text_feat,
                                torch.transpose(qcl_q_feat1_all, 1, 2)) / self.hparams.softmax_temperature
        qscores1 = torch.max(sim_mat1, dim=-1)[0]
        qscores1 = qscores1.t()
        qscores = (qscores0 + qscores1)/2

        sim_mat2 = torch.matmul(qcl_q_feat0,
                                qcl_text_feat_all.t()) / self.hparams.softmax_temperature
        qscores2 = torch.max(sim_mat2, dim=1)[0]
        sim_mat3 = torch.matmul(qcl_q_feat1,
                                qcl_text_feat_all.t()) / self.hparams.softmax_temperature
        qscores3 = torch.max(sim_mat3, dim=1)[0]
        qscores_t = (qscores2 + qscores3) / 2

        qloss0 = F.cross_entropy(qscores, cl_labels)
        qloss1 = F.cross_entropy(qscores_t, cl_labels)
        q_loss = qloss0 + qloss1

        self._dequeue_and_enqueue(cl_img_feat, cl_text_feat, qcl_text_feat, qcl_q_feat0.contiguous(), qcl_q_feat1.contiguous())

        #### Language Modeling ####

        q_feat = torch.cat([q_feat0, q_feat1], dim=1)
        q_feat = self.qformer.embed_1(q_feat)
        q_feat = F.normalize(q_feat, dim=-1)

        output = self.decoder(input_ids=batch['text0'], attention_mask=batch['attn0'],
                              encoder_hidden_states=q_feat.contiguous(), labels=batch['label0'])
        t_loss0 = output['loss']

        if batch['text1'] != None:
            output = self.decoder(input_ids=batch['text1'], attention_mask=batch['attn1'],
                               encoder_hidden_states=q_feat.contiguous(), labels=batch['label1'])
            t_loss1 = output['loss']
        else:
            t_loss1 = torch.tensor([0]).type_as(t_loss0)

        if batch['text2'] != None:
            output = self.decoder(input_ids=batch['text2'], attention_mask=batch['attn2'],
                               encoder_hidden_states=q_feat.contiguous(), labels=batch['label2'])
            t_loss2 = output['loss']

        else:
            t_loss2 = torch.tensor([0]).type_as(t_loss0)

        lm_loss = self.hparams.lambda_0 * t_loss0 + self.hparams.lambda_1 * t_loss1 + self.hparams.lambda_2 * t_loss2

        return c_loss, q_loss, lm_loss, acc1, acc5

        # # Forward of query image encoder
        # img_feat_q, patch_feat_q = self.img_encoder_q(
        #     batch["imgs"])  # CLS、patch编码特征
        # patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)  # 卷积线性映射+BN+AF+卷积线性映射+BN
        # patch_emb_q = F.normalize(patch_emb_q, dim=-1)  # 得到归一化特征
        # img_emb_q = self.img_encoder_q.global_embed(img_feat_q)  # 线性映射+BN+AF+线性映射+BN
        # img_emb_q = F.normalize(img_emb_q, dim=-1)  # 得到归一化特征
        #
        # # Forward of query text encoder
        # report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
        #     batch["caption_ids"], batch["attention_mask"], batch["position_ids"])
        # word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        # word_emb_q = F.normalize(word_emb_q, dim=-1)
        # report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        # report_emb_q = F.normalize(report_emb_q, dim=-1)  # 词编码特征归一化
        #
        # bz = img_emb_q.size(0)  # batch size
        # labels = torch.arange(bz).type_as(report_emb_q).long()  # 标签
        #
        # scores = img_emb_q.mm(report_emb_q.t())  # 全局信息求内积
        # scores /= self.hparams.softmax_temperature
        # scores1 = scores.transpose(0, 1)
        # loss0 = F.cross_entropy(scores, labels)  # 对角线上的图像-文本对是匹配的，得分应最高
        # loss1 = F.cross_entropy(scores1, labels)
        # loss_ita = loss0 + loss1  # 图像2文本损失+文本2图像损失
        #
        # # compute retrieval accuracy 计算k1正确率和k5正确率。并不要求对角线元素是最大的，只要是前k大的即可
        # i2t_acc1, i2t_acc5 = self.precision_at_k(
        #     scores, labels, top_k=(1, 5))  # 图像2文本
        # t2i_acc1, t2i_acc5 = self.precision_at_k(
        #     scores1, labels, top_k=(1, 5))  # 文本2图像
        # acc1 = (i2t_acc1 + t2i_acc1) / 2.
        # acc5 = (i2t_acc5 + t2i_acc5) / 2.

    # def sinkhorn(self, Q, nmb_iters):
    #     '''
    #         :param Q: (num_prototypes, batch size)
    #
    #     '''
    #     with torch.no_grad():
    #         sum_Q = torch.sum(Q)
    #         Q /= sum_Q  # 归一化
    #
    #         K, B = Q.shape  # K：原型数；B：bs
    #
    #         if self.hparams.gpus > 0:
    #             u = torch.zeros(K).cuda()
    #             r = torch.ones(K).cuda() / K
    #             c = torch.ones(B).cuda() / B
    #         else:
    #             u = torch.zeros(K)
    #             r = torch.ones(K) / K
    #             c = torch.ones(B) / B
    #
    #         for _ in range(nmb_iters):
    #             u = torch.sum(Q, dim=1)
    #             Q *= (r / u).unsqueeze(1)
    #             Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    #
    #         return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    # def distributed_sinkhorn(self, Q, nmb_iters):
    #     with torch.no_grad():
    #         sum_Q = torch.sum(Q)
    #         dist.all_reduce(sum_Q)
    #         Q /= sum_Q
    #
    #         if self.hparams.gpus > 0:
    #             u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
    #             r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
    #             c = torch.ones(Q.shape[1]).cuda(
    #                 non_blocking=True) / (self.gpus * Q.shape[1])
    #         else:
    #             u = torch.zeros(Q.shape[0])
    #             r = torch.ones(Q.shape[0]) / Q.shape[0]
    #             c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])
    #
    #         curr_sum = torch.sum(Q, dim=1)
    #         dist.all_reduce(curr_sum)
    #
    #         for it in range(nmb_iters):
    #             u = curr_sum
    #             Q *= (r / u).unsqueeze(1)
    #             Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    #             curr_sum = torch.sum(Q, dim=1)
    #             dist.all_reduce(curr_sum)
    #         return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text0_feat, text1_feat, qfeat0_feat, qfeat1_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text0_feats = concat_all_gather(text0_feat)
        text1_feats = concat_all_gather(text1_feat)
        qfeat0_feats = concat_all_gather(qfeat0_feat)
        qfeat1_feats = concat_all_gather(qfeat1_feat)
        # image_feats = image_feat
        # text0_feats = text0_feat
        # text1_feats = text1_feat
        # qfeat0_feats = qfeat0_feat
        # qfeat1_feats = qfeat1_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.img_queue[ptr:ptr + batch_size, :] = image_feats
        self.text0_queue[ptr:ptr + batch_size, :] = text0_feats
        self.text1_queue[ptr:ptr + batch_size, :] = text1_feats
        self.qfeat0_queue[ptr:ptr + batch_size, :, :] = qfeat0_feats
        self.qfeat1_queue[ptr:ptr + batch_size, :, :] = qfeat1_feats
        ptr = (ptr + batch_size) % self.hparams.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def encode(self, img0, img1):

        global_img0_feat, local_img0_feat = self.img_encoder_q(img0)
        global_img1_feat, local_img1_feat = self.img_encoder_q(img0)
        bs = len(global_img0_feat)

        lm_img0_feat = torch.cat([global_img0_feat.unsqueeze(1), local_img0_feat], dim=1)
        lm_img0_feat = self.img_encoder_q.embed_1(lm_img0_feat)
        lm_img0_feat = F.normalize(lm_img0_feat, dim=-1)
        lm_img1_feat = torch.cat([global_img1_feat.unsqueeze(1), local_img1_feat], dim=1)
        lm_img1_feat = self.img_encoder_q.embed_1(lm_img1_feat)
        lm_img1_feat = F.normalize(lm_img1_feat, dim=-1)

        qformer_input = torch.zeros([bs, 32, 768]).to(lm_img0_feat.device)
        global_q_feat0, local_q_feat0 = self.qformer(inputs_embeds=qformer_input, encoder_hidden_states=lm_img0_feat)
        q_feat0 = torch.cat([global_q_feat0.unsqueeze(1), local_q_feat0], dim=1)
        global_q_feat1, local_q_feat1 = self.qformer(inputs_embeds=qformer_input, encoder_hidden_states=lm_img1_feat)
        q_feat1 = torch.cat([global_q_feat1.unsqueeze(1), local_q_feat1], dim=1)

        q_feat = torch.cat([q_feat0, q_feat1], dim=1)
        q_feat = self.qformer.embed_1(q_feat)
        q_feat = F.normalize(q_feat, dim=-1)

        return q_feat

    def decode(self, input_ids, encoder_output):

        output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)

        return output


    def training_step(self, batch, batch_idx):
        # loss_ita, loss_local, loss_proto, acc1, acc5 = self(
            # batch, batch_idx, "train")
        # loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            # loss_local + self.hparams.lambda_3 * loss_proto
        c_loss, q_loss, lm_loss, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = lm_loss + q_loss + c_loss

        log = {
            "train_loss": loss,
            "train_contrastive_loss": c_loss,
            "train_qformer_contrastive_loss": q_loss,
            "train_language_model_loss": lm_loss,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    # def on_after_backward(self):
        # if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            # for param in self.prototype_layer.parameters():
                # param.grad = None

    def validation_step(self, batch, batch_idx):
        # loss_ita, loss_local, loss_proto, acc1, acc5 = self(
            # batch, batch_idx, "valid")  # 病例损失、词元损失、原型损失、病例损失的k1正确率、病例损失的k5正确率
        c_loss, q_loss, lm_loss, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = lm_loss + q_loss + c_loss

        log = {
            "val_loss": loss,
            "val_contrastive_loss": c_loss,
            "val_qformer_contrastive_loss": q_loss,
            "val_language_model_loss": lm_loss,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    # def on_train_epoch_end(self):
    #     ''' Save img_queue and report_queue for visualization '''
    #     if self.local_rank == 0:
    #         img_queue_path = f"{self.trainer.callbacks[-1].dirpath}/img_queue.pth"
    #         torch.save(self.img_queue, img_queue_path)
    #         report_queue_path = f"{self.trainer.callbacks[-1].dirpath}/report_queue.pth"
    #         torch.save(self.report_queue, report_queue_path)

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):  # 统计对角线元素是前k大元素的个数
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)  # 按行取最大元素，共取k次，得到一个(行数，k)的张量
            # 第一个参数是k，第二个参数的取最大的维度，0是取列最大，1是取行最大，第三个参数是取最大/最小，第四个参数是是否按序重排张量
            # 返回第一个变量是数值张量，第二个值是元素在原张量该行中的索引
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # 统计k1-k5的正确元素个数

            res = []
            for k in top_k:  # 计算k1正确率和k5正确率
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))  # 换算成正确率，除以batch再乘100
            return res

    def configure_optimizers(self):  # 模型优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps, # 训练步
            cycle_mult=1.0,  # 重启系数
            max_lr=self.hparams.learning_rate,  # 最大学习率
            min_lr=1e-8,  # 最小学习率
            warmup_steps=int(self.training_steps * 0.4)  # 线性warmup步数
        )  # 类似于模拟退火的重启学习率调整机制，在一定时间内会重启学习率，然后余弦下降，cycle_mult=1时固定间隔重启，>1时重启间隔随训练过程越来越长
        pass
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod  # 将一个方法转化为静态方法。静态方法不使用类的资源，可以直接调用而不实例化类，相当于类内的方法
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=25)
        # parser.add_argument("--num_prototypes", type=int, default=500)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_0", type=float, default=9)
        parser.add_argument("--lambda_1", type=float, default=1)
        parser.add_argument("--lambda_2", type=float, default=3)
        # parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--hidden_dim", type=int, default=768)
        parser.add_argument("--query_num", type=int, default=32)
        parser.add_argument("--queue_size", type=int, default=100)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()  # 检查多机分布式训练是否已经初始化

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices  # trainer.accumulate_grad_batches：每k个batch累计一次梯度

        return (dataset_size // effective_batch_size) * trainer.max_epochs

@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)  # 生成Trainer的参数，无需一个个手动赋值
    # model args
    parser = MGCA.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = MGCA(**args.__dict__)  # 加载backbone和预训练参数
    # model = MGCA.load_from_checkpoint('/home/sutongkun/VLPv2/MGCA/data/ckpts/MGCA/2023_09_05_17_37_25/epoch=39-step=5319.ckpt', strict=True)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())  # 当前时区时间
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")  # 变换时间格式
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/MGCA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"), # 按步记录学习率
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2), # 保存模型。monitor：按验证集损失保存模型；save_last：将最新的checkpoint保存为last.ckpt；mode：最小化问题；save_top_k：保存最优的5个模型
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")  # early stopping。min_delta：视为improvement的monitor变化量；patience：5次验证没有改善，则停止
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    trainer = Trainer.from_argparse_args(  # 初始化trainer
        args=args,
        callbacks=callbacks,
    )

    model.training_steps = model.num_training_steps(trainer, datamodule)  # 模型训练步数，训练步数指的是模型更新多少此权重
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)  # 模型训练

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)  # 保存最佳的k个模型到yaml文件中


if __name__ == "__main__":
    cli_main()
