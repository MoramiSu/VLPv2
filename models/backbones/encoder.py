import os

import torch
import torch.nn as nn
from einops import rearrange
from models.backbones import cnn_backbones
from models.backbones.med import BertModel
from models.backbones.vits import create_vit
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调转维度（B,F,N）
        x = self.head(x)

        return x.permute(0, 2, 1)  # 恢复维度顺序


class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        if "vit" in model_name:
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = model_name[4:]  # vit类型
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)  # 从url中加载vit参数
            state_dict = checkpoint["model"]
            msg = self.model.load_state_dict(state_dict, strict=False)  # 给vit加载预训练参数。strict表示是否严格根据state_dict键值和模型键值加载参数，msg返回参数加载是否匹配以及不匹配键值对

            self.embed_0 = GlobalEmbedding(
                vision_width, hidden_dim, output_dim
            )

            self.embed_1 = LocalEmbedding(
                vision_width, hidden_dim, vision_width
            )

        else:
            model_function = getattr(
                cnn_backbones, model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )

            # Average pooling
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 自适应平均池化，输出为1×1的矩阵

            self.embed_0 = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

            self.embed_1 = LocalEmbedding(
                self.interm_feature_dim, hidden_dim, 768
            )

    def resnet_forward(self, x, get_local=True):
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        return x, local_features.contiguous()

    def vit_forward(self, x):
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat = self.vit_forward(x)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()  # 返回编码后的CLS、词元特征，contiguous类似于copy


class BertEncoder(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = True,
                 split='text_encoder'
                 ):
        super(BertEncoder, self).__init__()
        # self.bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        # self.bert_type = "nghuyong/ernie-health-zh"
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True
        # self.max_sent_num = 10

        # 加载预训练模型
        if split == 'text_encoder':
            self.config = BertConfig.from_json_file(
                os.path.join(BASE_DIR, "../../configs/bert_config.json"))  # BERT的一些超参
            self.model = BertModel.from_pretrained(
                # self.bert_type,
                os.path.dirname(os.path.abspath(__file__)) + '/ERNIE',
                config=self.config,
                add_pooling_layer=False,
            )
            self.embed_0 = GlobalEmbedding(
                self.embedding_dim, hidden_dim, self.output_dim)  # 线性映射+BN+AF+线性映射+BN
            self.embed_1 = GlobalEmbedding(
                self.embedding_dim, hidden_dim, self.output_dim)  # 卷积线性映射+BN+AF+卷积线性映射+BN
        else:
            self.config = BertConfig.from_json_file(
                os.path.join(BASE_DIR, "../../configs/qformer_config.json"))  # BERT的一些超参
            # self.model = BertModel.from_pretrained(
            # # self.bert_type,
            #     os.path.dirname(os.path.abspath(__file__)) + '/GPT',
            #     config=self.config,
            #     add_pooling_layer=False,
            # )
            self.model = BertModel(self.config)

            self.embed_0 = LocalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)
            self.embed_1 = LocalEmbedding(self.embedding_dim, hidden_dim, self.embedding_dim)

        # if tokenizer:
        #     self.tokenizer = tokenizer
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        #
        # self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}  # idx2word dict

        if self.freeze_bert is True:  # 固定BERT的参数
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False
        # if split == 'text_encoder':
        #     self.embed_0 = GlobalEmbedding(
        #         self.embedding_dim, hidden_dim, self.output_dim)  # 线性映射+BN+AF+线性映射+BN
        #     self.embed_1 = GlobalEmbedding(
        #         self.embedding_dim, hidden_dim, self.output_dim)  # 卷积线性映射+BN+AF+卷积线性映射+BN
        # else:
        #     self.embed_0 = LocalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)
        #     self.embed_1 = LocalEmbedding(self.embedding_dim, hidden_dim, self.embedding_dim)

    # def aggregate_tokens(self, features, caption_ids):
    #     _, num_layers, num_words, dim = features.shape
    #     features = features.permute(0, 2, 1, 3)
    #     agg_feats_batch = []
    #     sentence_feats = []
    #     sentences = []
    #     sentence_mask = []

    #     # loop over batch
    #     for feats, caption_id in zip(features, caption_ids):
    #         agg_feats = []
    #         token_bank = []
    #         words = []
    #         word_bank = []
    #         sent_feat_list = []
    #         sent_feats = []
    #         sent_idx = 0

    #         # loop over sentence
    #         for i, (word_feat, word_id) in enumerate(zip(feats, caption_id)):
    #             word = self.idxtoword[word_id.item()]
    #             if word == "[PAD]":
    #                 new_feat = torch.stack(token_bank)
    #                 new_feat = new_feat.sum(axis=0)
    #                 agg_feats.append(new_feat)
    #                 words.append("".join(word_bank))

    #             elif word == "[SEP]":
    #                 new_feat = torch.stack(token_bank)
    #                 new_feat = new_feat.sum(axis=0)
    #                 agg_feats.append(new_feat)
    #                 sent_feat_list.append(new_feat)
    #                 words.append("".join(word_bank))

    #                 # if two consecutive SEP
    #                 if word_bank == ["[SEP]"]:
    #                     break

    #                 if i == num_words - 1:
    #                     # if it is the last word
    #                     agg_feats.append(word_feat)
    #                     words.append(word)
    #                 else:
    #                     # clear word features
    #                     token_bank = [word_feat]
    #                     word_bank = [word]

    #                 sent_feat_list.append(word_feat)

    #                 # aggregate sentence features
    #                 if sent_idx == 0:
    #                     # remove cls token
    #                     # use sum to aggragte token features
    #                     sent_feat = torch.stack(sent_feat_list[1:]).mean(dim=0)
    #                     # sent_feat = torch.stack(sent_feat_list[1:])
    #                 else:
    #                     sent_feat = torch.stack(sent_feat_list).mean(dim=0)
    #                     # sent_feat = torch.stack(sent_feat_list)

    #                 sent_feats.append(sent_feat)
    #                 # clear sent feat
    #                 sent_feat_list = []

    #                 # add sent_idx
    #                 sent_idx += 1

    #             # This is because some words are divided into two words.
    #             elif word.startswith("##"):
    #                 token_bank.append(word_feat)
    #                 word_bank.append(word[2:])

    #             else:
    #                 if len(word_bank) == 0:
    #                     token_bank.append(word_feat)
    #                     word_bank.append(word)
    #                 else:
    #                     new_feat = torch.stack(token_bank)
    #                     new_feat = new_feat.sum(axis=0)
    #                     agg_feats.append(new_feat)
    #                     # if not seq, add into sentence embeddings
    #                     if word_bank != ["[SEP]"]:
    #                         sent_feat_list.append(new_feat)
    #                     words.append("".join(word_bank))

    #                     token_bank = [word_feat]
    #                     word_bank = [word]

    #         agg_feats = torch.stack(agg_feats)
    #         padding_size = num_words - len(agg_feats)
    #         paddings = torch.zeros(
    #             padding_size, num_layers, dim).type_as(agg_feats)
    #         words = words + ["[PAD]"] * padding_size

    #         agg_feats_batch.append(torch.cat([agg_feats, paddings]))
    #         sentences.append(words)

    #         sent_len = min(len(sent_feats), self.max_sent_num)
    #         sent_mask = [False] * sent_len + [True] * \
    #             (self.max_sent_num - sent_len)
    #         sentence_mask.append(sent_mask)

    #         sent_feats = torch.stack(sent_feats)
    #         if len(sent_feats) >= self.max_sent_num:
    #             sentence_feats.append(sent_feats[:self.max_sent_num])
    #         else:
    #             padding_size = self.max_sent_num - len(sent_feats)
    #             paddings = torch.zeros(
    #                 padding_size, num_layers, dim).type_as(sent_feats)
    #             sentence_feats.append(torch.cat([sent_feats, paddings], dim=0))

    #     agg_feats_batch = torch.stack(agg_feats_batch)
    #     agg_feats_batch = agg_feats_batch.permute(0, 2, 1, 3)
    #     sentence_mask = torch.tensor(
    #         sentence_mask).type_as(agg_feats_batch).bool()
    #     sentence_feats = torch.stack(sentence_feats)
    #     sentence_feats = sentence_feats.permute(0, 2, 1, 3)

    #     return agg_feats_batch, sentence_feats, sentence_mask, sentences

    # def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
    #     '''
    #     :param embeddings: bz, 1, 112, 768
    #     :param caption_ids: bz, 112
    #     :param last_layer_attn: bz, 111
    #     '''
    #     _, num_layers, num_words, dim = embeddings.shape
    #     embeddings = embeddings.permute(0, 2, 1, 3)  # B，N，1，feat
    #     agg_embs_batch = []
    #     sentences = []
    #     last_attns = []
    #
    #     # loop over batch
    #     for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):  # 遍历batch中的每个文本
    #         agg_embs = []  # 词元融合后的词嵌入
    #         token_bank = []  # 词元嵌入库
    #         words = []  # 词元融合后得到的词
    #         word_bank = []  # 词元库
    #         attns = []  # 词元融合后CLS对其注意力得分
    #         attn_bank = []  # CLS对词元注意力得分库
    #
    #         # loop over sentence
    #         for word_emb, word_id, attn in zip(embs, caption_id, last_attn):  # 对于一个文本中的每个词元
    #             word = self.idxtoword[word_id.item()]  # idx2word
    #             if word == "[SEP]":  # 句末
    #                 # 词元融合
    #                 new_emb = torch.stack(token_bank)
    #                 new_emb = new_emb.sum(axis=0)
    #                 agg_embs.append(new_emb)
    #                 words.append("".join(word_bank))
    #                 attns.append(sum(attn_bank))
    #                 agg_embs.append(word_emb)
    #                 words.append(word)
    #                 attns.append(attn)
    #                 break
    #             # This is because some words are divided into two words.部分单词被划分成多个词元，需将其嵌入和得分进行修正
    #             if not word.startswith("##"):  # 当前词元是否与之前的词元成词
    #                 if len(word_bank) == 0:  # 第一个词元的初始化
    #                     token_bank.append(word_emb)
    #                     word_bank.append(word)
    #                     attn_bank.append(attn)
    #                 else:  # 不成词
    #                     new_emb = torch.stack(token_bank)
    #                     new_emb = new_emb.sum(axis=0)  # 将词元嵌入库中存放的所有词元嵌入求和，作为这些词元对应词的嵌入
    #                     agg_embs.append(new_emb)
    #                     words.append("".join(word_bank))  # 拼接词元嵌入库中的所有词元，构成一个词
    #                     attns.append(sum(attn_bank))  # 将CLS对库中各词元的注意力得分之和作为CLS对库中词元构成的词的注意力得分
    #
    #                     # 库清空并装入新词元
    #                     token_bank = [word_emb]
    #                     word_bank = [word]
    #                     attn_bank = [attn]
    #             else:  # 成词
    #                 # 入库
    #                 token_bank.append(word_emb)
    #                 word_bank.append(word[2:])
    #                 attn_bank.append(attn)
    #         # 进行填充，以构成batch
    #         agg_embs = torch.stack(agg_embs)
    #         padding_size = num_words - len(agg_embs)  # 需要填充词元数
    #         paddings = torch.zeros(padding_size, num_layers, dim)  # 构造词嵌入的填充
    #         paddings = paddings.type_as(agg_embs)
    #         words = words + ["[PAD]"] * padding_size  # 词填充
    #         last_attns.append(
    #             torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))  # 词注意力得分填充
    #         agg_embs_batch.append(torch.cat([agg_embs, paddings]))  # 词嵌入填充
    #         sentences.append(words)
    #
    #     agg_embs_batch = torch.stack(agg_embs_batch)
    #     agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)  # B，1，N，feat
    #     last_atten_pt = torch.stack(last_attns)
    #     last_atten_pt = last_atten_pt.type_as(agg_embs_batch)  # 类型转换
    #
    #     return agg_embs_batch, sentences, last_atten_pt

    def forward(self, ids=None, attn_mask=None, position_ids=None, get_local=False, encoder_hidden_states=None, inputs_embeds=None):
        if encoder_hidden_states != None:
            mode = 'multimodal'
        else:
            mode = 'text'
        outputs = self.model(input_ids=ids, inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states,
                             return_dict=True, mode=mode)

        # last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)  # 最后一层中CLS词元对其他词元的各头注意力得分的平均，或理解为CLS词元对其他词元的注意力得分
        # last_layer_attn = outputs.attentions[-1][:, :, 0, :].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)  # 词元编码特征

        # if self.agg_tokens:  # 针对部分单纯被划分为多个词元，通过词元融合将词元的嵌入、注意力得分转化为单词的嵌入和注意力得分
        #     all_feat, sents, last_atten_pt = self.aggregate_tokens(
        #         all_feat, ids, last_layer_attn)
        #     last_atten_pt = last_atten_pt[:, 1:].contiguous()
        # else:
        #     sents = [[self.idxtoword[w.item()] for w in sent]
        #              for sent in ids]

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]  # 压掉unsqueeze增加的一维

        report_feat = all_feat[:, 0].contiguous()  # 全局特征
        word_feat = all_feat[:, 1:].contiguous()  # 词特征

        # sent_feat = all_feat.mean(axis=2)
        # if self.aggregate_method == "sum":
        #     word_feat = all_feat.sum(axis=1)
        #     sent_feat = sent_feat.sum(axis=1)
        # elif self.aggregate_method == "mean":
        #     word_feat = all_feat.mean(axis=1)
        #     sent_feat = sent_feat.mean(axis=1)
        # else:
        #     print(self.aggregate_method)
        #     raise Exception("Aggregation method not implemented")

        # aggregate intermetidate layers
        # TODO: try to remove word later
        # if self.last_n_layers > 1:
        #     all_feat = torch.stack(
        #         all_feat[-self.last_n_layers:]
        #     )  # layers, batch, sent_len, embedding size

        #     all_feat = all_feat.permute(1, 0, 2, 3)

        #     if self.agg_tokens:
        #         all_feat, sents = self.aggregate_tokens(all_feat, ids)
        #     else:
        #         sents = [[self.idxtoword[w.item()] for w in sent]
        #                  for sent in ids]
        #     sent_feat = all_feat.mean(axis=2)

        #     if self.aggregate_method == "sum":
        #         word_feat = all_feat.sum(axis=1)
        #         sent_feat = sent_feat.sum(axis=1)
        #     elif self.aggregate_method == "mean":
        #         word_feat = all_feat.mean(axis=1)
        #         sent_feat = sent_feat.mean(axis=1)
        #     else:
        #         print(self.aggregate_method)
        #         raise Exception("Aggregation method not implemented")
        # else:
        #     # use last layer
        #     word_feat, sent_feat = outputs[0], outputs[1]
        # word_feat = rearrange(word_feat, "b n d -> b d n")

        # if get_local:
        #     return word_feat, sents, sent_indices
        # else:
        #     return sents, sent_indices

        # report_feat, report_atten_weights = self.atten_pooling(sent_feat)

        # use cls token as report features
        # report_feat = word_feat[:, 0].contiguous()
        # use mean here

        return report_feat, word_feat


if __name__ == "__main__":
    from MGCA.datasets.pretrain_dataset import MultimodalPretrainingDataset
    from MGCA.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)

    for i, data in enumerate(dataset):
        imgs, caps, cap_len, key = data
        # if caps["attention_mask"].sum() == 112:
        if caps["attention_mask"].sum() == 150:
            model = BertEncoder()
            report_feat, sent_feat, sent_mask, sents = model(
                caps["input_ids"],
                caps["attention_mask"],
                caps["token_type_ids"],
                get_local=True)
            print(report_feat, sent_feat, sent_mask, sents)
            break
