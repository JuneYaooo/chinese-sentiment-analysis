
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from utils import processing
import pandas as pd


# 超参数
embed_size = 100
hidden_size = 64
num_layers = 2
# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 需先搭建网络模型model
# 网络结构
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向, 输出维度要*2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 双向, 第一个维度要*2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        
        packed_out, (h_n, h_c) = self.lstm(packed_input, (h0, c0))

        lstm_out = torch.cat([h_n[-2], h_n[-1]], 1)  # 双向, 所以要将最后两维拼接, 得到的就是最后一个time step的输出
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out

lstm = LSTM(embed_size, hidden_size, num_layers).to(device)   
# 然后通过下面的语句加载参数
lstm=torch.load('/root/nas/chinese-sentiment-analysis/model/classification/lstm_10k_7.model')


import time
from gensim import models
t1 = time.time()
word2vec = models.Word2Vec.load('/root/nas/chinese-sentiment-analysis/model/word/word2vec_10k.model')
t2 = time.time()
print(".molde load time %.4f"%(t2-t1))


def collate_fn(data):
    """
    :param data: 第0维：data，第1维：label
    :return: 序列化的data、记录实际长度的序列、以及label列表
    """
    data.sort(key=lambda x: len(x[0]), reverse=True) # pack_padded_sequence要求要按照序列的长度倒序排列
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    data = pad_sequence(x, batch_first=True, padding_value=0)   # 用RNN处理变长序列的必要操作
    return data, torch.tensor(y, dtype=torch.float32), torch.tensor(data_length)


def calculate_emo_score(text):
    vectors= []
    for w in processing(text).split(" "):
        if w in word2vec.wv.key_to_index:
            vectors.append(word2vec.wv[w])   # 将每个词替换为对应的词向量
    vectors = torch.Tensor(vectors)
    x, _, lengths = collate_fn(list(zip([vectors], [-1])))
    if lengths[0].item()<1:
        return None
    with torch.no_grad():
        x = x.to(device)
        outputs = lstm(x, lengths)       # 前向传播
        outputs = outputs.view(-1)    # 将输出展平
        result_score = outputs.cpu()[0].item()      
    return result_score

# 验证
crawl_result = pd.read_csv('/root/nas/comment-crawler/crawl_result.csv')
crawl_result['emo_score'] = crawl_result['texts'].apply(lambda x: calculate_emo_score(x))
crawl_result.to_csv('/root/nas/chinese-sentiment-analysis/result/crawl_result_emo.csv', index=False, encoding='utf-8-sig')