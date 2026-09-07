from transformers import PretrainedConfig
from typing import Optional
import math 

#PretrainedConfig是Hugging Face Transformers库中的一个基类，用于定义预训练模型的配置。它包含了模型的超参数和其他相关信息，
# 允许用户在加载或保存模型时使用一致的配置。通过继承PretrainedConfig，用户可以创建自定义模型配置类，以便在训练和推理过程中使用特定的参数设置。
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

'''
RMSnorm 的作用就是归一化,防止计算值过大或者过小导致数值不稳定,从而加速模型的训练和收敛.
它通过计算输入的均值和标准差来进行归一化,并且引入了一个可学习的缩放参数来调整归一化后的输出.
相比于 LayerNorm, RMSNorm 不需要计算均值,因此在某些情况下可以更高效地进行归一化.
通常防止梯度爆炸和梯度消失,从而提高模型的训练稳定性和性能.
'''
#继承nn.Module类
import torch
import torch.nn as nn
class RMSNorm(nn.Module):
#__init__初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))#Parameter会自动标注为可训练参数,并且会被加入到模型的参数列表中,在训练过程中会被优化器更新.
#norm
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
#forward前向传播
    def forward(self,x):
        return x*self._norm(x.float()).type_as(x) * self.weight.type_as(x)
def precompute_freqs_cli(dim:int,rope_base,end:int=32*1024,rope_scaling:Optional[dict]=None):
    #初始化rope频率
    freqs,attn_factor = (1.0/(rope_base ** (torch.arange(0,dim,2).float()/dim)),1.0)
    #如果rope_scaling不为空,则进行缩放
    if rope_scaling is not None:
        original_max,factor,beta_fast,beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"])
        #推断长度大于原始最大位置嵌入长度时,进行缩放
        if end>original_max:
            #频率b到i的映射
            inv_dim = lambda b:(dim*math.log(original_max/(b*2*math.pi)))/(2*math.log(rope_base))#b本质上就是该维度的归一化频率（Normalized Frequency），它代表着这个维度在整个上下文窗口内振荡的总圈数（周期数）
            #划分高低维度
            #low:不需要缩放的高频部分
            #high:需要缩放的低频部分
            low,high = (max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))

            #计算缩放因子
            #在low之前，ramp为0,在high之后，ramp为1,在low和high之间，ramp为线性插值，ramp保存了每个维度的缩放因子
            ramp = torch.clamp(
                (torch.arange(dim // 2,device=freqs.device).float()-low)
                / max(high-low,0.001),
                0,
                1
            )
            freqs = freqs * (1-ramp+ramp/factor)
        # 根据end，生成位置索引
        t = torch.arange(end,device=freqs.device).float()

        #计算外积，将t和缩放因子相乘，得到每个位置的旋转角度
        freqs = torch.outer(t,freqs).float()
        freqs_cos = (
            torch.cat(
                [torch.cos(freqs),torch.cos(freqs)],
                dim=-1
            )*attn_factor
        )
        freqs_sin = (
            torch.cat(
                [torch.sin(freqs),torch.sin(freqs)],
                dim=-1
            )*attn_factor
        )
        return freqs_cos, freqs_sin
#编写rope
def apply_rotary_pos_emb(q,k,freqs_cos,freqs_sin):
    #[a,b]->[-b,a]
    def rotate_half(x):
        return torch.cat([-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],dim=-1).reshape_as(x)
    #实现x_rotated = x * cos + rotate_half(x) * sin
    q_embed = (q*freqs_cos.unsqueeze(1))+rotate_half(q)*freqs_sin.unsqueeze(1)
    k_embed = (k*freqs_cos.unsqueeze(1))+rotate_half(k)*freqs_sin.unsqueeze(1)#unsqueeze
    return q_embed, k_embed
