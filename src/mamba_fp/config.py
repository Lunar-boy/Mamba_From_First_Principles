from dataclasses import dataclass

@dataclass
class Mamba2Config:
    d_model: int = 256 # 隐藏层维度D
    n_layer: int = 4 # 层数L
    vocab_size: int = 1000 # 词汇表大小V
    d_state: int = 64 # 状态维度S
    d_conv: int = 64 # 因果卷积核
    expand: int = 2 #内部扩展倍数E d_inner = expand * d_model
    headdim: int = 64 # 每个头的维度P d_inner % headdim == 0
    chunk_size: int = 32 # SSD分块大小Q cpu32 H100用256
    ngroups: int = 1 # B 和 C的分组数，类似GQA
    bias: bool = False # 是否使用偏置
    conv_bias: bool = True # 是否使用卷积偏置 

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nhead(self) -> int:
        assert self.d_inner % self.headdim == 0
        return self.d_inner // self.headdim