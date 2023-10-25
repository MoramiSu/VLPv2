"""
Manage beam search info structure.
Heavily borrowed from OpenNMT-py.
For code in OpenNMT-py, please check the following link (maybe in oldest version):
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch

class Constants():
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.BOS_WORD = '[CLS]'
        self.EOS_WORD = '[SEP]'

    @classmethod
    def from_tokenizer(cls, tokenizer):  # cls=Constance
        instance = cls()
        instance.PAD = tokenizer.vocab[instance.PAD_WORD]
        instance.UNK = tokenizer.vocab[instance.UNK_WORD]
        instance.BOS = tokenizer.vocab[instance.BOS_WORD]
        instance.EOS = tokenizer.vocab[instance.EOS_WORD]
        return instance

class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False, tokenizer=None, prompt=None):
        if tokenizer is None:
            self.constants = Constants()
        else:
            self.constants = Constants.from_tokenizer(tokenizer)  # 特殊词元的idx

        self.size = size  # beam size
        self._done = False  # 解码完成标志
        # The score for each interface on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)  # 各beam得分
        self.all_scores = []  # 每一步解码的得分

        if prompt == None:
            # The backpointers at each time-step.
            self.prev_ks = []  # 记录每一步解码时最好的结果来自于哪些beam
            # The outputs at each time-step.
            self.next_ys = [torch.full((size,), self.constants.BOS, dtype=torch.long, device=device)]  # 记录当前解码结果。初始化时则构造一个所有元素均为self.constants.BOS的张量作为解码器的初始输入
        else:
            self.next_ys = []
            self.prev_ks = []
            for item in prompt:
                self.prev_ks.append(list(range(size)))
                self.next_ys.append(torch.full((size, ), item, dtype=torch.long, device=device))
            self.prev_ks = self.prev_ks[:-1]
    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()  # 得到当前各beam预测结果

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob, word_length=None):

        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)  # 词汇量
        # Sum the previous scores.
        if len(self.prev_ks) > 0:  # ？
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)  # 新预测的每个词的得分+之前预测的得分，即预测的得分之和
        else:  # 第一次解码
            beam_lk = word_prob[0]  # 第一个beam的预测结果，因为其他四个beam的预测结果相同
        flat_beam_lk = beam_lk.view(-1)  # 拉成1维
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort  # 在所有beam中搜索得分最高的五个分数和索引
        self.all_scores.append(self.scores)  # 加入上一次预测的得分
        self.scores = best_scores  # 记录预测的得分之和
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words  # 得分最高的预测分别来自哪些beam，由于view拉成1维，得到一个beam size*num_words的一维张量，因此除掉num_words取整，就可以得到各个元素来源于哪个beam
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)  # 加上本次预测的词索引
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.constants.EOS:  # 若预测概率最大的词是EOS，则解码完成
            self._done = True

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)  # 根据解码的得分之和排序

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:  # 第一次解码
            dec_seq = self.next_ys[0].unsqueeze(1)  # 增加一个维度
        else:
            _, keys = self.sort_scores()  # 预测得分从大到小的beam索引
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):  # 回溯得到beam k的词序列
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])  # 得到该beam的词
            k = self.prev_ks[j][k]  # 得到该beam从哪个beam迁移而来

        return list(map(lambda x: x.item(), hyp[::-1]))  # 回溯得到的词序列是逆序的，将其转化为顺序
