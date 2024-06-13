# Author:SiZhen
# Create: 2024/6/12
# Description: pytorch构建一个transformer框架
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from  collections import namedtuple
from torch.autograd import Variable

#克隆n次
def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module)for _ in range(n)])

class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)


class TextEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

#层归一化
class LayerNorm(nn.Module):
    def __init__(self,feature,eps=1e-6):
        super(LayerNorm, self).__init__()
        #feature 是self-attention中x的大小
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps #epsilon,一个很小的正数，用来避免除以零或者其他数值稳定性问题。

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/math.sqrt(std+self.eps)  + self.b_2

#残差和层归一化
class SublayerConnection(nn.Module):
   def __init__(self,size,dropout= 0.1):
       super(SublayerConnection, self).__init__()
       #层归一化
       self.layer_norm = LayerNorm(size)
       #随机失活
       self.dropout = nn.Dropout(p=dropout)

   def forward(self,x,sublayer):
       #x:self-attention的输入，sublayer:self-attention层
       return self.dropout(self.layer_norm(x+sublayer(x)))

#自注意力机制
def self_attention(query,key,value,dropout=None,mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask==0,-1e9)
    self_attn = F.softmax(scores,dim=-1)
    if dropout is not None :
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn,value),self_attn

#多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,head,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        #d_model是输入的维度
        assert (d_model % head == 0)
        self.d_k = d_model // head #每个头分配到的维度数，空间上并行学习，增加模型的表达能力
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model,d_model)
        self.linear_key = nn.Linear(d_model,d_model)
        self.linear_value = nn.Linear(d_model,d_model)
        # 自注意力机制的QKV同源，线性变换
        self.linear_out = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)#需要对X切分成多头

        query = self.linear_query(query).view(n_batch,-1,self.head,self.d_k).tranpose(1,2) #[b,8,32,64]
        key = self.linear_key(key).view(n_batch,-1,self.head,self.d_k).tranpose(1,2) #[b,8,28,64]
        value = self.linear_value(value).view(n_batch,-1,self.head,self.d_k).tranpose(1,2) #[b,8,28,64]


        x,self.attn = self_attention(query,key,value,dropout=self.dropout,mask=mask)
        x = x.transpose(1,2).contiguous().view(n_batch,-1,self.head*self.d_k) #[b,32*8,512]

        return self.linear_out(x)

#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,dim,dropout,max_len=5000):
        super(PositionalEncoding, self).__init__()
        if dim % 2 !=0:
            raise ValueError("Cannot use sin/cos positional encoding with"
                             "odd dim (got dim = {:d})".format(dim))
        #位置编码pe : PE(pos,2i/2i+1) = sin/cos (pos/10000^{2i/d_{model}})
        pe = torch.zeros(max_len,dim) #max_len是解码器生成句子的最长的长度，假设是10
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0,dim,2,dtype=torch.float)*
                             -(math.log(10000.0)/dim)))

        pe[:,0::2]=torch.sin(position.float()*div_term)
        pe[:,1::2] = torch.cos(position.float()*div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe',pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self,emb,step = None):
        #emb:初始的x
        emb = emb*math.sqrt(self.dim)
        if step is None :
            emb = emb+self.pe[:emb.size(0)]
        else:
            emb = emb+self.pe[step]
        emb = self.drop_out(emb)
        return emb

#前馈神经网络feedforward
class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self,x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output

#Linear和softmax
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.linear(x),dim=-1)

#掩码自注意力机制
def pad_mask(src,r2l_trg,trg,pad_idx):
    if isinstance(src,tuple):
        if len(src)==4:
            src_image_mask = (src[0][:,:,0]!=pad_idx).unsqueeze(1)
            src_motion_mask=(src[1][:,:,0]!=pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:,:,0]!=pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask=(src_image_mask,src_motion_mask,src_object_mask,src_rel_mask)
            dec_src_mask_1=src_image_mask & src_motion_mask
            dec_src_mask_2=src_image_mask & src_motion_mask &src_object_mask & src_rel_mask
            dec_src_mask =(dec_src_mask_1,dec_src_mask_2)
            src_mask = (enc_src_mask,dec_src_mask)
        if len(src)==3:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
            dec_src_mask = src_image_mask &src_motion_mask
            src_mask = (enc_src_mask,dec_src_mask)
        if len(src)==2:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask &src_motion_mask
            src_mask = (enc_src_mask,dec_src_mask)
    else:
        src_mask = (src[:,:,0]!= pad_idx).unsqueeze(1)
    if trg is not None:
        if isinstance(src_mask,tuple):
            trg_mask = subsequent_mask(trg.size(1)).type_as(src_image_mask.data)
            trg_mask = (trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)


    else:
        return src_mask


def subsequent_mask(size):
    attn_shape = (1,size,size)
    mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return (torch.from_numpy(mask)==0).cuda()

#编码器模块
class EncoderLayer(nn.Module):
    def __init__(self,size,attn,feed_forward,dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size,dropout),2)

    def forward(self,x,mask):
        x = self.sublayer_connection[0](x,lambda x:self.attn(x,x,x,mask))
        return self.sublayer_connection[1](x,self.feed_forward)

#整个编码器
class Encoder(nn.Module):
    def __init__(self,n,encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer,n)

    def forward(self,x,src_mask):
        for layer in self.encoder_layer:
            x = layer(x,src_mask)
        return x

#解码器模块
class DecoderLayer(nn.Module):
    def __init__(self,size,attn,feed_forward,sublayer_num,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feedforward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size,dropout),sublayer_num)

    def forward(self,x,memory,src_mask,trg_mask,r2l_memory=None,r2l_trg_mask=None):
        x = self.sublayer_connection[0](x,lambda x:self.attn(x,x,x,trg_mask))
        x = self.sublayer_connection[1](x,lambda x:self.attn(x,memory,memory,src_mask))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x,lambda x:self.attn(x,r2l_memory,r2l_memory,r2l_trg_mask))
        return self.sublayer_connection[-1](x,self.feedforward)

#双向解码器（右到左）
class R2L_Decoder(nn.Module):
    def __init__(self,n,decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer,n)
    def forward(self,x,memory,src_mask,r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x,memory,src_mask,r2l_trg_mask)
        return  x

#双向解码器(左到右)
class L2R_Decoder(nn.Module):
    def __init__(self,n,decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer,n)

    def forward(self,x,memory,src_mask,trg_mask,r2l_memory,r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x,memory,src_mask,trg_mask,r2l_memory,r2l_trg_mask)
        return x

#构建Transformer
class ABDTransformer(nn.Module):
    def __init__(self,vocab,d_feat,d_model,d_ff,n_heads,n_layers,dropout,feature_mode,
                 device = 'cuda',n_heads_big=128):
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode #多模态

        c = copy.deepcopy

        attn_no_heads = MultiHeadAttention(0,d_model,dropout)
        attn = MultiHeadAttention(n_heads,d_model,dropout)
        attn_big = MultiHeadAttention(n_heads_big,d_model,dropout)

        feed_forward = PositionWiseFeedForward(d_model,d_ff)
        if feature_mode == 'one':
            self.src_embed = FeatEmbedding(d_feat,d_model,dropout)
        elif feature_mode == "two":
            self.image_src_embed= FeatEmbedding(d_feat[0],d_model,dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1],d_model,dropout)
        elif feature_mode == 'three':
            self.image_src_embed = FeatEmbedding(d_feat[0],d_model,dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1],d_model,dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2].d_model,dropout)
        elif feature_mode == 'four':
            self.image_src_embed = FeatEmbedding(d_feat[0],d_model,dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1],d_model,dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2],d_model,dropout)
            self.rel_src_embed = FeatEmbedding(d_feat[3],d_model,dropout)

        self.trg_embed = TextEmbedding(vocab.n_vocabs,d_model)
        self.pos_embed = PositionalEncoding(d_model,dropout)
        self.encoder = Encoder(n_layers,Encoder(d_model,c(attn),c(feed_forward),dropout))

        self.r2l_decoder = R2L_Decoder(n_layers,DecoderLayer(d_model,c(attn),c(feed_forward),
                                                             sublayer_num=3,dropout=dropout))
        self.l2r_decoder = L2R_Decoder(n_layers,DecoderLayer(d_model,c(attn),c(feed_forward),
                                       sublayer_num=4,dropout=dropout))
        self.generator = Generator(d_model,vocab.n_vocabs)

    def encode(self, src, src_mask, feature_mode_two=False):
        if self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        if feature_mode_two:
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        if self.feature_mode == 'one':
            x = self.src_embed(src)
            x = self.pos_embed(x)
            return self.encoder(x, src_mask)
        elif self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        elif self.feature_mode == 'three':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])
            x3 = self.object_src_embed(src[2])
            x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            return x1 + x2 + x3
        elif self.feature_mode == 'four':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])

            x3 = self.object_src_embed(src[2])
            # x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            # x3 = self.encoder_no_attention(x3, src_mask[2])

            x4 = self.rel_src_embed(src[3])
            # x4 = self.pos_embed(x4)
            # x4 = self.encoder_no_
            # heads(x4, src_mask[3])
            x4 = self.encoder_no_attention(x4, src_mask[3])
            # x4 = self.encoder(x4, src_mask[3])
            return x1 + x2 + x3 + x4

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        if self.feature_mode == 'one':
            encoding_outputs = self.encode(src, src_mask)
            r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        elif self.feature_mode == 'two' or 'three' or 'four':
            enc_src_mask, dec_src_mask = src_mask
            r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
            encoding_outputs = self.encode(src, enc_src_mask)

            r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask[0], r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask[1], trg_mask, r2l_outputs, r2l_pad_mask)

            # r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, dec_src_mask, r2l_trg_mask)
            # l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, None, None)
        else:
            raise Exception("没有输出")

        r2l_pred = self.generator(r2l_outputs)
        l2r_pred = self.generator(l2r_outputs)

        return r2l_pred, l2r_pred











