import os
import datetime
from dateutil import tz

import numpy as np
import torch
import torch.nn as nn
from models.backbones.module_decoder import DecoderModel
from models.backbones.beam import Beam
from pytorch_lightning import LightningModule
from nlgeval import NLGEval
from transformers import BertTokenizer

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Captioner(LightningModule):
    def __init__(self,
                 encoder,
                 word_embedding_weight,
                 positional_embedding_weight,
                 decoder_config,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.Config = decoder_config
        self.encoder = encoder
        self.decoder = DecoderModel(decoder_config, word_embedding_weight, positional_embedding_weight)
        self.decoder.embeddings.word_embeddings.weight = word_embedding_weight
        self.decoder.embeddings.position_embeddings.weight = positional_embedding_weight
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.NLGEval = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)
        self.decoder_loss_fcn = nn.CrossEntropyLoss()
        self.output_dir = '/home/sutongkun/Pretrain_VLP_Project/MGCA_Ultrasonic/MGCA/data/caption'
        self.lr = learning_rate
        self.weight_decay = weight_decay

        now = datetime.datetime.now(tz.tzlocal())
        extension = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.hyp_path = os.path.join(self.output_dir, f"{extension}_hyp.txt")
        self.ref_path = os.path.join(self.output_dir, f"{extension}_ref.txt")

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.embeddings.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.train()
        img = batch['imgs']
        input_caption_ids = batch['input_caption_ids']
        attention_mask = batch['attention_mask']
        output_caption_ids = batch['output_caption_ids']
        bs = len(img)
        device = img.device
        encoder_mask = torch.ones((bs, 196)).to(device)

        _, feat = self.encoder(img)
        decoder_scores = self.decoder(input_caption_ids, feat, attention_mask, encoder_mask)
        loss = self.decoder_loss_fcn(decoder_scores.view(-1, self.Config.vocab_size),
                                             output_caption_ids.view(-1))

        self.log("train_loss", loss.item(), on_epoch=True, on_step=False, logger=True, prog_bar=True)

        return {'loss': loss, 'score': decoder_scores}

    def share_step(self, batch, split):
        self.encoder.eval()
        self.decoder.eval()
        all_result_lists = []
        all_caption_lists = []
        img = batch['imgs']
        input_caption_ids = batch['input_caption_ids']
        attention_mask = batch['attention_mask']
        output_caption_ids = batch['output_caption_ids']
        bs = len(img)

        with torch.no_grad():
            _, feat = self.encoder(img)

            n_bm = 5  # beam_size
            device = img.device

            feat = feat.repeat(1, n_bm, 1).view(bs * n_bm, 196, 768)
            encoder_mask = torch.ones((bs * n_bm, 196)).to(device)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=self.tokenizer) for _ in range(bs)]  # 对每个batch建立beam对象
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(bs))  # 记录还未解码完成的batch
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)  # idx2batch映射关系，表征数据的每一行对应哪一个batch，因为各batch完成解码的时间点不一样，先完成解码的batch会被删除，后面的batch补上
            # -- Decode
            for len_dec_seq in range(1, self.Config.max_words + 1):  # 对第一个词进行解码、第二个词进行解码，以此类推
                active_inst_idx_list = beam_decode_step(self.decoder, inst_dec_beams,
                                                            len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                            feat, encoder_mask)  # 解码+beam search，得到解码还未完成的batch

                if not active_inst_idx_list:  # 每个batch都解码完成
                    break  # all instances have finished their path to <EOS>

                feat, encoder_mask, inst_idx_to_position_map = collate_active_info(feat, encoder_mask, inst_idx_to_position_map, active_inst_idx_list, n_bm, device)  # 删除已完成解码的batch及其数据，保留未完成解码的batch，建立新的映射

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)  # 得到每个batch最好的解码得分和结果
            result_list = [batch_hyp[i][0] for i in range(bs)]  # 解码结果

            caption_list = output_caption_ids.cpu().detach().numpy()

                # 将解码结果转回文字
            for re_idx, re_list in enumerate(result_list):
                decode_text_list = self.tokenizer.convert_ids_to_tokens(re_list)  # 转回文本
                if "[SEP]" in decode_text_list:  # 若解码结果中有[SEP]词元，则取[SEP]前的文本
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:  # 若解码结果中有[PAD]词元，则取[PAD]前的文本
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()  # 去掉##符号
                all_result_lists.append(decode_text)

            # 将ground truth转回文字
            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = self.tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_caption_lists.append(decode_text)

        if split == 'test':
            # Save pure results
            with open(self.hyp_path, "a", encoding='utf-8') as writer:
                for pre_txt in all_result_lists:
                    writer.write(pre_txt.replace(' ', '') + "\n")

            with open(self.ref_path, "a", encoding='utf-8') as writer:
                for ground_txt in all_caption_lists:
                    writer.write(ground_txt.replace(' ', '') + "\n")

        # Evaluate
        metrics_nlg = self.NLGEval.compute_metrics(ref_list=[all_caption_lists], hyp_list=all_result_lists)
        self.log('BLEU_1', metrics_nlg['Bleu_1'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('BLEU_2', metrics_nlg['Bleu_2'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('BLEU_3', metrics_nlg['Bleu_3'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('BLEU_4', metrics_nlg['Bleu_4'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('METEOR', metrics_nlg['METEOR'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('ROUGE_L', metrics_nlg['ROUGE_L'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('CIDEr', metrics_nlg['CIDEr'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        return {'res': all_result_lists, 'ref': all_caption_lists}

    def test_step(self, batch, batch_idx):
        return self.share_step(batch, 'test')

    def validation_step(self, batch, batch_idx):
        return self.share_step(batch, 'valid')
    '''
    def training_epoch_end(self, outputs):
        ids = torch.argmax(outputs[-1]['score'], dim=2)
        ids = ids[0].cpu().numpy().tolist()
        decode_text_list = self.tokenizer.convert_ids_to_tokens(ids)
        print(decode_text_list)


    def validation_epoch_end(self, outputs):
        print(outputs[-1]['res'][0])
        print('/')
        print(outputs[-1]['ref'][0])
        print('/')
        print(outputs[-1]['res'][1])
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs

def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm, device, encoder_output, encoder_mask, decoder_length=None):


    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]  # 当前每个未完成解码的beam的预测结果
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)  # 合并batch size和beam size
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, encoder_output, encoder_mask):
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)  # 当前解码器输入的mask，由于完成解码的batch已被删除，剩下的数据都是未完成解码的batch，不存在padding，全置1即可
        next_decoder_ids = next_decoder_ids.view(-1, next_decoder_ids.shape[-1])
        next_decoder_mask = next_decoder_mask.view(-1, next_decoder_mask.shape[-1])
        dec_output = decoder(next_decoder_ids, encoder_output, next_decoder_mask, encoder_mask) # 将next_decoder_ids和next_decoder_mask作为input_caption_id和input_caption_mask参与下一个词的解码
        dec_output = dec_output[:, -1, :]  # 压掉第二个维度
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)  # 将batch size和beam size拆开
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():  # 对各个未完成解码的batch进行beam search
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])  # 返回解码是否完成
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:  # 该batch解码是否完成
                active_inst_idx_list += [inst_idx]  # 记录解码还未完成的batch

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)  # 未解码完成的batch数
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)  # 当前未完成解码的每个beam的预测结果
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, encoder_output, encoder_mask)  # 下一个词的预测结果

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)  # beam search，得到解码还未完成的batch

    return active_inst_idx_list

def collate_active_info(encoder_output, encoder_mask, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)  # batch size
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]  # 解码还未完成的batch
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_encoder_output = collect_active_part(encoder_output, active_inst_idx, n_prev_active_inst, n_bm)  # 保留解码未完成batch的数据，删除解码已完成的batch的数据
    active_encoder_mask = collect_active_part(encoder_mask, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)  # 得到idx2batch的映射关系（删除已完成的batch）

    return active_encoder_output, active_encoder_mask, active_inst_idx_to_position_map

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()  # d_hs：后两个维度size组装起来的list
    n_curr_active_inst = len(curr_active_inst_idx)  # 解码未完成的batch数
    new_shape = (n_curr_active_inst * n_bm, *d_hs)  # 删除已完成解码数据后的形状

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)  # 合并后两个维度
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)  # 保留解码未完成的batch的数据，删除解码已完成的batch数据
    beamed_tensor = beamed_tensor.view(*new_shape)  # 还原

    return beamed_tensor

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):  # 遍历所有的batch
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()  # 当前batch所有beam的得分从大到小排序，返回得分和对应的beam索引
        all_scores += [scores[:n_best]]  # 记录最好的n_best个得分

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]  # 得到最好的n_best个得分对应的词序列
        all_hyp += [hyps]
    return all_hyp, all_scores