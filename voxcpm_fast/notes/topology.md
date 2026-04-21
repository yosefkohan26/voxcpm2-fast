# VoxCPM2 Topology

- arch: `VoxCPM2Model`
- total params: `2,290,004,544`
- feat_dim: `64`  patch_size: `4`
- inference_timesteps: `10`

## Module tree

```
base_lm: Cpm4Model  params=1,471,744,000
  embed_tokens: VocabParallelEmbedding  params=150,421,504
  layers: ModuleList  params=1,321,320,448
    0: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    1: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    2: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    3: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    4: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    5: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    6: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    7: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    8: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    9: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    10: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    11: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    12: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    13: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    14: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    15: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    16: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    17: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    18: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    19: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    20: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    21: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    22: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    23: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    24: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    25: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    26: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    27: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        rotary_emb: MiniCPMLongRoPE  params=0
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
  norm: RMSNorm  params=2,048
residual_lm: Cpm4Model  params=377,522,176
  embed_tokens: Identity  params=0
  layers: ModuleList  params=377,520,128
    0: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    1: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    2: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    3: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    4: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    5: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    6: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
    7: Cpm4DecoderLayer  params=47,190,016
      self_attn: Cpm4Attention  params=9,437,184
        qkv_proj: QKVParallelLinear  params=5,242,880
        o_proj: RowParallelLinear  params=4,194,304
        attn: Attention  params=0
      mlp: Cpm4MLP  params=37,748,736
        gate_up_proj: MergedColumnParallelLinear  params=25,165,824
        down_proj: RowParallelLinear  params=12,582,912
        act_fn: SiluAndMul  params=0
      input_layernorm: RMSNorm  params=2,048
      post_attention_layernorm: RMSNorm  params=2,048
  norm: RMSNorm  params=2,048
feat_encoder: VoxCPM2LocEnc  params=207,711,232
  in_proj: Linear  params=66,560
  encoder: Cpm4Model  params=207,643,648
    embed_tokens: Identity  params=0
    layers: ModuleList  params=207,642,624
      0: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      1: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      2: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      3: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      4: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      5: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      6: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      7: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      8: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      9: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      10: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
      11: Cpm4DecoderLayer  params=17,303,552
        self_attn: Cpm4Attention  params=4,718,592
          qkv_proj: QKVParallelLinear  params=2,621,440
          o_proj: RowParallelLinear  params=2,097,152
          rotary_emb: MiniCPMLongRoPE  params=0
          attn: Attention  params=0
        mlp: Cpm4MLP  params=12,582,912
          gate_up_proj: MergedColumnParallelLinear  params=8,388,608
          down_proj: RowParallelLinear  params=4,194,304
          act_fn: SiluAndMul  params=0
        input_layernorm: RMSNorm  params=1,024
        post_attention_layernorm: RMSNorm  params=1,024
    norm: RMSNorm  params=1,024
feat_decoder: UnifiedCFM  params=212,040,768
  estimator: VoxCPM2LocDiT  params=212,040,768
    in_proj: Linear  params=66,560
    cond_proj: Linear  params=66,560
    out_proj: Linear  params=65,600
    time_embeddings: SinusoidalPosEmb  params=0
    time_mlp: TimestepEmbedding  params=2,099,200
      linear_1: Linear  params=1,049,600
      act: SiLU  params=0
      linear_2: Linear  params=1,049,600
    delta_time_mlp: TimestepEmbedding  params=2,099,200
      linear_1: Linear  params=1,049,600
      act: SiLU  params=0
      linear_2: Linear  params=1,049,600
    decoder: Cpm4Model  params=207,643,648
      embed_tokens: Identity  params=0
      layers: ModuleList  params=207,642,624
        0: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        1: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        2: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        3: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        4: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        5: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        6: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        7: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        8: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        9: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        10: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
        11: Cpm4DecoderLayer  params=17,303,552
          self_attn: Cpm4Attention  params=4,718,592
            qkv_proj: QKVParallelLinear  params=2,621,440
            o_proj: RowParallelLinear  params=2,097,152
            rotary_emb: MiniCPMLongRoPE  params=0
            attn: Attention  params=0
          mlp: Cpm4MLP  params=12,582,912
            gate_up_proj: MergedColumnParallelLinear  params=8,388,608
            down_proj: RowParallelLinear  params=4,194,304
            act_fn: SiluAndMul  params=0
          input_layernorm: RMSNorm  params=1,024
          post_attention_layernorm: RMSNorm  params=1,024
      norm: RMSNorm  params=1,024
fsq_layer: ScalarQuantizationLayer  params=2,099,712
  in_proj: Linear  params=1,049,088
  out_proj: Linear  params=1,050,624
enc_to_lm_proj: Linear  params=2,099,200
lm_to_dit_proj: Linear  params=2,098,176
res_to_dit_proj: Linear  params=2,098,176
fusion_concat_proj: Linear  params=8,390,656
stop_proj: Linear  params=4,196,352
stop_actn: SiLU  params=0
stop_head: Linear  params=4,096
```

## Forward order (one decode step, concurrency=1)

Total module calls: `2029`

| # | module | in shapes | out shapes |
|---|---|---|---|
| 0 | `feat_encoder.in_proj` | `((1, 4, 64),)` | `(1, 4, 1024)` |
| 1 | `feat_encoder.encoder.layers.0.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 2 | `feat_encoder.encoder.layers.0.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 3 | `feat_encoder.encoder.layers.0.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 4 | `feat_encoder.encoder.layers.0.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 5 | `feat_encoder.encoder.layers.0.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 6 | `feat_encoder.encoder.layers.0.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 7 | `feat_encoder.encoder.layers.0.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 8 | `feat_encoder.encoder.layers.0.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 9 | `feat_encoder.encoder.layers.0.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 10 | `feat_encoder.encoder.layers.0.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 11 | `feat_encoder.encoder.layers.0.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 12 | `feat_encoder.encoder.layers.0` | `((5,), (1, 5, 1024), 'NoneType')` | `((1, 5, 1024), (1, 5, 1024))` |
| 13 | `feat_encoder.encoder.layers.1.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 14 | `feat_encoder.encoder.layers.1.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 15 | `feat_encoder.encoder.layers.1.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 16 | `feat_encoder.encoder.layers.1.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 17 | `feat_encoder.encoder.layers.1.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 18 | `feat_encoder.encoder.layers.1.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 19 | `feat_encoder.encoder.layers.1.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 20 | `feat_encoder.encoder.layers.1.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 21 | `feat_encoder.encoder.layers.1.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 22 | `feat_encoder.encoder.layers.1.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 23 | `feat_encoder.encoder.layers.1.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 24 | `feat_encoder.encoder.layers.1` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 25 | `feat_encoder.encoder.layers.2.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 26 | `feat_encoder.encoder.layers.2.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 27 | `feat_encoder.encoder.layers.2.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 28 | `feat_encoder.encoder.layers.2.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 29 | `feat_encoder.encoder.layers.2.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 30 | `feat_encoder.encoder.layers.2.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 31 | `feat_encoder.encoder.layers.2.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 32 | `feat_encoder.encoder.layers.2.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 33 | `feat_encoder.encoder.layers.2.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 34 | `feat_encoder.encoder.layers.2.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 35 | `feat_encoder.encoder.layers.2.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 36 | `feat_encoder.encoder.layers.2` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 37 | `feat_encoder.encoder.layers.3.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 38 | `feat_encoder.encoder.layers.3.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 39 | `feat_encoder.encoder.layers.3.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 40 | `feat_encoder.encoder.layers.3.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 41 | `feat_encoder.encoder.layers.3.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 42 | `feat_encoder.encoder.layers.3.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 43 | `feat_encoder.encoder.layers.3.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 44 | `feat_encoder.encoder.layers.3.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 45 | `feat_encoder.encoder.layers.3.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 46 | `feat_encoder.encoder.layers.3.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 47 | `feat_encoder.encoder.layers.3.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 48 | `feat_encoder.encoder.layers.3` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 49 | `feat_encoder.encoder.layers.4.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 50 | `feat_encoder.encoder.layers.4.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 51 | `feat_encoder.encoder.layers.4.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 52 | `feat_encoder.encoder.layers.4.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 53 | `feat_encoder.encoder.layers.4.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 54 | `feat_encoder.encoder.layers.4.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 55 | `feat_encoder.encoder.layers.4.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 56 | `feat_encoder.encoder.layers.4.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 57 | `feat_encoder.encoder.layers.4.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 58 | `feat_encoder.encoder.layers.4.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 59 | `feat_encoder.encoder.layers.4.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 60 | `feat_encoder.encoder.layers.4` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 61 | `feat_encoder.encoder.layers.5.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 62 | `feat_encoder.encoder.layers.5.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 63 | `feat_encoder.encoder.layers.5.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 64 | `feat_encoder.encoder.layers.5.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 65 | `feat_encoder.encoder.layers.5.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 66 | `feat_encoder.encoder.layers.5.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 67 | `feat_encoder.encoder.layers.5.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 68 | `feat_encoder.encoder.layers.5.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 69 | `feat_encoder.encoder.layers.5.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 70 | `feat_encoder.encoder.layers.5.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 71 | `feat_encoder.encoder.layers.5.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 72 | `feat_encoder.encoder.layers.5` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 73 | `feat_encoder.encoder.layers.6.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 74 | `feat_encoder.encoder.layers.6.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 75 | `feat_encoder.encoder.layers.6.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 76 | `feat_encoder.encoder.layers.6.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 77 | `feat_encoder.encoder.layers.6.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 78 | `feat_encoder.encoder.layers.6.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 79 | `feat_encoder.encoder.layers.6.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 80 | `feat_encoder.encoder.layers.6.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 81 | `feat_encoder.encoder.layers.6.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 82 | `feat_encoder.encoder.layers.6.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 83 | `feat_encoder.encoder.layers.6.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 84 | `feat_encoder.encoder.layers.6` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 85 | `feat_encoder.encoder.layers.7.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 86 | `feat_encoder.encoder.layers.7.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 87 | `feat_encoder.encoder.layers.7.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 88 | `feat_encoder.encoder.layers.7.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 89 | `feat_encoder.encoder.layers.7.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 90 | `feat_encoder.encoder.layers.7.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 91 | `feat_encoder.encoder.layers.7.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 92 | `feat_encoder.encoder.layers.7.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 93 | `feat_encoder.encoder.layers.7.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 94 | `feat_encoder.encoder.layers.7.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 95 | `feat_encoder.encoder.layers.7.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 96 | `feat_encoder.encoder.layers.7` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 97 | `feat_encoder.encoder.layers.8.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 98 | `feat_encoder.encoder.layers.8.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 99 | `feat_encoder.encoder.layers.8.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 100 | `feat_encoder.encoder.layers.8.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 101 | `feat_encoder.encoder.layers.8.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 102 | `feat_encoder.encoder.layers.8.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 103 | `feat_encoder.encoder.layers.8.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 104 | `feat_encoder.encoder.layers.8.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 105 | `feat_encoder.encoder.layers.8.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 106 | `feat_encoder.encoder.layers.8.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 107 | `feat_encoder.encoder.layers.8.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 108 | `feat_encoder.encoder.layers.8` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 109 | `feat_encoder.encoder.layers.9.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 110 | `feat_encoder.encoder.layers.9.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 111 | `feat_encoder.encoder.layers.9.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 112 | `feat_encoder.encoder.layers.9.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 113 | `feat_encoder.encoder.layers.9.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 114 | `feat_encoder.encoder.layers.9.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 115 | `feat_encoder.encoder.layers.9.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 116 | `feat_encoder.encoder.layers.9.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 117 | `feat_encoder.encoder.layers.9.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 118 | `feat_encoder.encoder.layers.9.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 119 | `feat_encoder.encoder.layers.9.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 120 | `feat_encoder.encoder.layers.9` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 121 | `feat_encoder.encoder.layers.10.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 122 | `feat_encoder.encoder.layers.10.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 123 | `feat_encoder.encoder.layers.10.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 124 | `feat_encoder.encoder.layers.10.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 125 | `feat_encoder.encoder.layers.10.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 126 | `feat_encoder.encoder.layers.10.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 127 | `feat_encoder.encoder.layers.10.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 128 | `feat_encoder.encoder.layers.10.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 129 | `feat_encoder.encoder.layers.10.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 130 | `feat_encoder.encoder.layers.10.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 131 | `feat_encoder.encoder.layers.10.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 132 | `feat_encoder.encoder.layers.10` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 133 | `feat_encoder.encoder.layers.11.input_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 134 | `feat_encoder.encoder.layers.11.self_attn.qkv_proj` | `((1, 5, 1024),)` | `(1, 5, 2560)` |
| 135 | `feat_encoder.encoder.layers.11.self_attn.rotary_emb` | `((5,), (1, 5, 2048), (1, 5, 256))` | `((1, 5, 2048), (1, 5, 256))` |
| 136 | `feat_encoder.encoder.layers.11.self_attn.attn` | `((1, 5, 16, 128), (1, 5, 2, 128), (1, 5, 2, 128))` | `(1, 5, 16, 128)` |
| 137 | `feat_encoder.encoder.layers.11.self_attn.o_proj` | `((1, 5, 2048),)` | `(1, 5, 1024)` |
| 138 | `feat_encoder.encoder.layers.11.self_attn` | `((5,), (1, 5, 1024))` | `(1, 5, 1024)` |
| 139 | `feat_encoder.encoder.layers.11.post_attention_layernorm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 140 | `feat_encoder.encoder.layers.11.mlp.gate_up_proj` | `((1, 5, 1024),)` | `(1, 5, 8192)` |
| 141 | `feat_encoder.encoder.layers.11.mlp.act_fn` | `((1, 5, 8192),)` | `(1, 5, 4096)` |
| 142 | `feat_encoder.encoder.layers.11.mlp.down_proj` | `((1, 5, 4096),)` | `(1, 5, 1024)` |
| 143 | `feat_encoder.encoder.layers.11.mlp` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 144 | `feat_encoder.encoder.layers.11` | `((5,), (1, 5, 1024), (1, 5, 1024))` | `((1, 5, 1024), (1, 5, 1024))` |
| 145 | `feat_encoder.encoder.norm` | `((1, 5, 1024),)` | `(1, 5, 1024)` |
| 146 | `feat_encoder.encoder` | `((1, 5, 1024), (5,))` | `(1, 5, 1024)` |
| 147 | `feat_encoder` | `((1, 4, 64),)` | `(1, 1024)` |
| 148 | `enc_to_lm_proj` | `((1, 1024),)` | `(1, 2048)` |
| 149 | `base_lm.embed_tokens` | `((1,),)` | `(1, 2048)` |
| 150 | `base_lm.layers.0.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 151 | `base_lm.layers.0.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 152 | `base_lm.layers.0.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 153 | `base_lm.layers.0.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 154 | `base_lm.layers.0.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 155 | `base_lm.layers.0.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 156 | `base_lm.layers.0.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 157 | `base_lm.layers.0.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 158 | `base_lm.layers.0.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 159 | `base_lm.layers.0.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 160 | `base_lm.layers.0.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 161 | `base_lm.layers.0` | `((1,), (1, 2048), 'NoneType')` | `((1, 2048), (1, 2048))` |
| 162 | `base_lm.layers.1.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 163 | `base_lm.layers.1.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 164 | `base_lm.layers.1.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 165 | `base_lm.layers.1.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 166 | `base_lm.layers.1.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 167 | `base_lm.layers.1.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 168 | `base_lm.layers.1.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 169 | `base_lm.layers.1.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 170 | `base_lm.layers.1.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 171 | `base_lm.layers.1.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 172 | `base_lm.layers.1.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 173 | `base_lm.layers.1` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 174 | `base_lm.layers.2.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 175 | `base_lm.layers.2.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 176 | `base_lm.layers.2.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 177 | `base_lm.layers.2.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 178 | `base_lm.layers.2.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 179 | `base_lm.layers.2.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 180 | `base_lm.layers.2.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 181 | `base_lm.layers.2.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 182 | `base_lm.layers.2.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 183 | `base_lm.layers.2.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 184 | `base_lm.layers.2.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 185 | `base_lm.layers.2` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 186 | `base_lm.layers.3.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 187 | `base_lm.layers.3.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 188 | `base_lm.layers.3.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 189 | `base_lm.layers.3.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 190 | `base_lm.layers.3.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 191 | `base_lm.layers.3.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 192 | `base_lm.layers.3.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 193 | `base_lm.layers.3.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 194 | `base_lm.layers.3.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 195 | `base_lm.layers.3.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 196 | `base_lm.layers.3.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 197 | `base_lm.layers.3` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 198 | `base_lm.layers.4.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 199 | `base_lm.layers.4.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 200 | `base_lm.layers.4.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 201 | `base_lm.layers.4.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 202 | `base_lm.layers.4.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 203 | `base_lm.layers.4.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 204 | `base_lm.layers.4.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 205 | `base_lm.layers.4.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 206 | `base_lm.layers.4.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 207 | `base_lm.layers.4.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 208 | `base_lm.layers.4.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 209 | `base_lm.layers.4` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 210 | `base_lm.layers.5.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 211 | `base_lm.layers.5.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 212 | `base_lm.layers.5.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 213 | `base_lm.layers.5.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 214 | `base_lm.layers.5.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 215 | `base_lm.layers.5.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 216 | `base_lm.layers.5.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 217 | `base_lm.layers.5.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 218 | `base_lm.layers.5.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 219 | `base_lm.layers.5.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 220 | `base_lm.layers.5.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 221 | `base_lm.layers.5` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 222 | `base_lm.layers.6.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 223 | `base_lm.layers.6.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 224 | `base_lm.layers.6.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 225 | `base_lm.layers.6.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 226 | `base_lm.layers.6.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 227 | `base_lm.layers.6.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 228 | `base_lm.layers.6.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 229 | `base_lm.layers.6.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 230 | `base_lm.layers.6.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 231 | `base_lm.layers.6.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 232 | `base_lm.layers.6.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 233 | `base_lm.layers.6` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 234 | `base_lm.layers.7.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 235 | `base_lm.layers.7.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 236 | `base_lm.layers.7.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 237 | `base_lm.layers.7.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 238 | `base_lm.layers.7.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 239 | `base_lm.layers.7.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 240 | `base_lm.layers.7.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 241 | `base_lm.layers.7.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 242 | `base_lm.layers.7.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 243 | `base_lm.layers.7.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 244 | `base_lm.layers.7.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 245 | `base_lm.layers.7` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 246 | `base_lm.layers.8.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 247 | `base_lm.layers.8.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 248 | `base_lm.layers.8.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 249 | `base_lm.layers.8.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 250 | `base_lm.layers.8.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 251 | `base_lm.layers.8.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 252 | `base_lm.layers.8.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 253 | `base_lm.layers.8.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 254 | `base_lm.layers.8.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 255 | `base_lm.layers.8.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 256 | `base_lm.layers.8.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 257 | `base_lm.layers.8` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 258 | `base_lm.layers.9.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 259 | `base_lm.layers.9.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 260 | `base_lm.layers.9.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 261 | `base_lm.layers.9.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 262 | `base_lm.layers.9.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 263 | `base_lm.layers.9.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 264 | `base_lm.layers.9.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 265 | `base_lm.layers.9.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 266 | `base_lm.layers.9.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 267 | `base_lm.layers.9.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 268 | `base_lm.layers.9.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 269 | `base_lm.layers.9` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 270 | `base_lm.layers.10.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 271 | `base_lm.layers.10.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 272 | `base_lm.layers.10.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 273 | `base_lm.layers.10.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 274 | `base_lm.layers.10.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 275 | `base_lm.layers.10.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 276 | `base_lm.layers.10.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 277 | `base_lm.layers.10.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 278 | `base_lm.layers.10.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 279 | `base_lm.layers.10.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 280 | `base_lm.layers.10.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 281 | `base_lm.layers.10` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 282 | `base_lm.layers.11.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 283 | `base_lm.layers.11.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 284 | `base_lm.layers.11.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 285 | `base_lm.layers.11.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 286 | `base_lm.layers.11.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 287 | `base_lm.layers.11.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 288 | `base_lm.layers.11.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 289 | `base_lm.layers.11.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 290 | `base_lm.layers.11.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 291 | `base_lm.layers.11.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 292 | `base_lm.layers.11.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 293 | `base_lm.layers.11` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 294 | `base_lm.layers.12.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 295 | `base_lm.layers.12.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 296 | `base_lm.layers.12.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 297 | `base_lm.layers.12.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 298 | `base_lm.layers.12.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 299 | `base_lm.layers.12.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 300 | `base_lm.layers.12.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 301 | `base_lm.layers.12.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 302 | `base_lm.layers.12.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 303 | `base_lm.layers.12.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 304 | `base_lm.layers.12.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 305 | `base_lm.layers.12` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 306 | `base_lm.layers.13.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 307 | `base_lm.layers.13.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 308 | `base_lm.layers.13.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 309 | `base_lm.layers.13.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 310 | `base_lm.layers.13.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 311 | `base_lm.layers.13.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 312 | `base_lm.layers.13.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 313 | `base_lm.layers.13.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 314 | `base_lm.layers.13.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 315 | `base_lm.layers.13.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 316 | `base_lm.layers.13.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 317 | `base_lm.layers.13` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 318 | `base_lm.layers.14.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 319 | `base_lm.layers.14.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 320 | `base_lm.layers.14.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 321 | `base_lm.layers.14.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 322 | `base_lm.layers.14.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 323 | `base_lm.layers.14.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 324 | `base_lm.layers.14.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 325 | `base_lm.layers.14.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 326 | `base_lm.layers.14.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 327 | `base_lm.layers.14.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 328 | `base_lm.layers.14.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 329 | `base_lm.layers.14` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 330 | `base_lm.layers.15.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 331 | `base_lm.layers.15.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 332 | `base_lm.layers.15.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 333 | `base_lm.layers.15.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 334 | `base_lm.layers.15.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 335 | `base_lm.layers.15.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 336 | `base_lm.layers.15.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 337 | `base_lm.layers.15.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 338 | `base_lm.layers.15.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 339 | `base_lm.layers.15.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 340 | `base_lm.layers.15.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 341 | `base_lm.layers.15` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 342 | `base_lm.layers.16.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 343 | `base_lm.layers.16.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 344 | `base_lm.layers.16.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 345 | `base_lm.layers.16.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 346 | `base_lm.layers.16.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 347 | `base_lm.layers.16.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 348 | `base_lm.layers.16.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 349 | `base_lm.layers.16.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 350 | `base_lm.layers.16.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 351 | `base_lm.layers.16.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 352 | `base_lm.layers.16.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 353 | `base_lm.layers.16` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 354 | `base_lm.layers.17.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 355 | `base_lm.layers.17.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 356 | `base_lm.layers.17.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 357 | `base_lm.layers.17.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 358 | `base_lm.layers.17.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 359 | `base_lm.layers.17.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 360 | `base_lm.layers.17.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 361 | `base_lm.layers.17.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 362 | `base_lm.layers.17.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 363 | `base_lm.layers.17.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 364 | `base_lm.layers.17.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 365 | `base_lm.layers.17` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 366 | `base_lm.layers.18.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 367 | `base_lm.layers.18.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 368 | `base_lm.layers.18.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 369 | `base_lm.layers.18.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 370 | `base_lm.layers.18.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 371 | `base_lm.layers.18.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 372 | `base_lm.layers.18.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 373 | `base_lm.layers.18.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 374 | `base_lm.layers.18.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 375 | `base_lm.layers.18.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 376 | `base_lm.layers.18.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 377 | `base_lm.layers.18` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 378 | `base_lm.layers.19.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 379 | `base_lm.layers.19.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 380 | `base_lm.layers.19.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 381 | `base_lm.layers.19.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 382 | `base_lm.layers.19.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 383 | `base_lm.layers.19.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 384 | `base_lm.layers.19.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 385 | `base_lm.layers.19.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 386 | `base_lm.layers.19.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 387 | `base_lm.layers.19.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 388 | `base_lm.layers.19.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 389 | `base_lm.layers.19` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 390 | `base_lm.layers.20.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 391 | `base_lm.layers.20.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 392 | `base_lm.layers.20.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 393 | `base_lm.layers.20.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 394 | `base_lm.layers.20.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 395 | `base_lm.layers.20.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 396 | `base_lm.layers.20.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 397 | `base_lm.layers.20.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 398 | `base_lm.layers.20.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 399 | `base_lm.layers.20.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 400 | `base_lm.layers.20.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 401 | `base_lm.layers.20` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 402 | `base_lm.layers.21.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 403 | `base_lm.layers.21.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 404 | `base_lm.layers.21.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 405 | `base_lm.layers.21.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 406 | `base_lm.layers.21.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 407 | `base_lm.layers.21.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 408 | `base_lm.layers.21.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 409 | `base_lm.layers.21.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 410 | `base_lm.layers.21.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 411 | `base_lm.layers.21.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 412 | `base_lm.layers.21.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 413 | `base_lm.layers.21` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 414 | `base_lm.layers.22.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 415 | `base_lm.layers.22.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 416 | `base_lm.layers.22.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 417 | `base_lm.layers.22.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 418 | `base_lm.layers.22.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 419 | `base_lm.layers.22.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 420 | `base_lm.layers.22.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 421 | `base_lm.layers.22.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 422 | `base_lm.layers.22.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 423 | `base_lm.layers.22.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 424 | `base_lm.layers.22.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 425 | `base_lm.layers.22` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 426 | `base_lm.layers.23.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 427 | `base_lm.layers.23.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 428 | `base_lm.layers.23.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 429 | `base_lm.layers.23.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 430 | `base_lm.layers.23.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 431 | `base_lm.layers.23.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 432 | `base_lm.layers.23.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 433 | `base_lm.layers.23.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 434 | `base_lm.layers.23.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 435 | `base_lm.layers.23.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 436 | `base_lm.layers.23.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 437 | `base_lm.layers.23` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 438 | `base_lm.layers.24.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 439 | `base_lm.layers.24.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 440 | `base_lm.layers.24.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 441 | `base_lm.layers.24.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 442 | `base_lm.layers.24.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 443 | `base_lm.layers.24.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 444 | `base_lm.layers.24.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 445 | `base_lm.layers.24.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 446 | `base_lm.layers.24.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 447 | `base_lm.layers.24.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 448 | `base_lm.layers.24.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 449 | `base_lm.layers.24` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 450 | `base_lm.layers.25.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 451 | `base_lm.layers.25.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 452 | `base_lm.layers.25.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 453 | `base_lm.layers.25.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 454 | `base_lm.layers.25.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 455 | `base_lm.layers.25.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 456 | `base_lm.layers.25.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 457 | `base_lm.layers.25.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 458 | `base_lm.layers.25.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 459 | `base_lm.layers.25.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 460 | `base_lm.layers.25.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 461 | `base_lm.layers.25` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 462 | `base_lm.layers.26.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 463 | `base_lm.layers.26.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 464 | `base_lm.layers.26.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 465 | `base_lm.layers.26.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 466 | `base_lm.layers.26.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 467 | `base_lm.layers.26.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 468 | `base_lm.layers.26.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 469 | `base_lm.layers.26.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 470 | `base_lm.layers.26.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 471 | `base_lm.layers.26.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 472 | `base_lm.layers.26.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 473 | `base_lm.layers.26` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 474 | `base_lm.layers.27.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 475 | `base_lm.layers.27.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 476 | `base_lm.layers.27.self_attn.rotary_emb` | `((1,), (1, 2048), (1, 256))` | `((1, 2048), (1, 256))` |
| 477 | `base_lm.layers.27.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 478 | `base_lm.layers.27.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 479 | `base_lm.layers.27.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 480 | `base_lm.layers.27.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 481 | `base_lm.layers.27.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 482 | `base_lm.layers.27.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 483 | `base_lm.layers.27.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 484 | `base_lm.layers.27.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 485 | `base_lm.layers.27` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 486 | `base_lm.norm` | `((1, 2048),)` | `(1, 2048)` |
| 487 | `base_lm` | `((1, 2048), (1,))` | `(1, 2048)` |
| 488 | `fsq_layer.in_proj` | `((1, 2048),)` | `(1, 512)` |
| 489 | `fsq_layer.out_proj` | `((1, 512),)` | `(1, 2048)` |
| 490 | `fsq_layer` | `((1, 2048),)` | `(1, 2048)` |
| 491 | `fusion_concat_proj` | `((1, 4096),)` | `(1, 2048)` |
| 492 | `residual_lm.layers.0.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 493 | `residual_lm.layers.0.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 494 | `residual_lm.layers.0.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 495 | `residual_lm.layers.0.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 496 | `residual_lm.layers.0.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 497 | `residual_lm.layers.0.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 498 | `residual_lm.layers.0.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 499 | `residual_lm.layers.0.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 500 | `residual_lm.layers.0.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 501 | `residual_lm.layers.0.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 502 | `residual_lm.layers.0` | `((1,), (1, 2048), 'NoneType')` | `((1, 2048), (1, 2048))` |
| 503 | `residual_lm.layers.1.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 504 | `residual_lm.layers.1.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 505 | `residual_lm.layers.1.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 506 | `residual_lm.layers.1.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 507 | `residual_lm.layers.1.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 508 | `residual_lm.layers.1.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 509 | `residual_lm.layers.1.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 510 | `residual_lm.layers.1.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 511 | `residual_lm.layers.1.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 512 | `residual_lm.layers.1.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 513 | `residual_lm.layers.1` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 514 | `residual_lm.layers.2.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 515 | `residual_lm.layers.2.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 516 | `residual_lm.layers.2.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 517 | `residual_lm.layers.2.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 518 | `residual_lm.layers.2.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 519 | `residual_lm.layers.2.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 520 | `residual_lm.layers.2.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 521 | `residual_lm.layers.2.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 522 | `residual_lm.layers.2.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 523 | `residual_lm.layers.2.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 524 | `residual_lm.layers.2` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 525 | `residual_lm.layers.3.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 526 | `residual_lm.layers.3.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 527 | `residual_lm.layers.3.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 528 | `residual_lm.layers.3.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 529 | `residual_lm.layers.3.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 530 | `residual_lm.layers.3.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 531 | `residual_lm.layers.3.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 532 | `residual_lm.layers.3.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 533 | `residual_lm.layers.3.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 534 | `residual_lm.layers.3.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 535 | `residual_lm.layers.3` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 536 | `residual_lm.layers.4.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 537 | `residual_lm.layers.4.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 538 | `residual_lm.layers.4.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 539 | `residual_lm.layers.4.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 540 | `residual_lm.layers.4.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 541 | `residual_lm.layers.4.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 542 | `residual_lm.layers.4.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 543 | `residual_lm.layers.4.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 544 | `residual_lm.layers.4.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 545 | `residual_lm.layers.4.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 546 | `residual_lm.layers.4` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 547 | `residual_lm.layers.5.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 548 | `residual_lm.layers.5.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 549 | `residual_lm.layers.5.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 550 | `residual_lm.layers.5.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 551 | `residual_lm.layers.5.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 552 | `residual_lm.layers.5.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 553 | `residual_lm.layers.5.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 554 | `residual_lm.layers.5.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 555 | `residual_lm.layers.5.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 556 | `residual_lm.layers.5.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 557 | `residual_lm.layers.5` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 558 | `residual_lm.layers.6.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 559 | `residual_lm.layers.6.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 560 | `residual_lm.layers.6.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 561 | `residual_lm.layers.6.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 562 | `residual_lm.layers.6.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 563 | `residual_lm.layers.6.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 564 | `residual_lm.layers.6.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 565 | `residual_lm.layers.6.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 566 | `residual_lm.layers.6.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 567 | `residual_lm.layers.6.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 568 | `residual_lm.layers.6` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 569 | `residual_lm.layers.7.input_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 570 | `residual_lm.layers.7.self_attn.qkv_proj` | `((1, 2048),)` | `(1, 2560)` |
| 571 | `residual_lm.layers.7.self_attn.attn` | `((1, 16, 128), (1, 2, 128), (1, 2, 128))` | `(1, 1, 16, 128)` |
| 572 | `residual_lm.layers.7.self_attn.o_proj` | `((1, 2048),)` | `(1, 2048)` |
| 573 | `residual_lm.layers.7.self_attn` | `((1,), (1, 2048))` | `(1, 2048)` |
| 574 | `residual_lm.layers.7.post_attention_layernorm` | `((1, 2048),)` | `(1, 2048)` |
| 575 | `residual_lm.layers.7.mlp.gate_up_proj` | `((1, 2048),)` | `(1, 12288)` |
| 576 | `residual_lm.layers.7.mlp.act_fn` | `((1, 12288),)` | `(1, 6144)` |
| 577 | `residual_lm.layers.7.mlp.down_proj` | `((1, 6144),)` | `(1, 2048)` |
| 578 | `residual_lm.layers.7.mlp` | `((1, 2048),)` | `(1, 2048)` |
| 579 | `residual_lm.layers.7` | `((1,), (1, 2048), (1, 2048))` | `((1, 2048), (1, 2048))` |
| 580 | `residual_lm.norm` | `((1, 2048),)` | `(1, 2048)` |
| 581 | `residual_lm` | `((1, 2048), (1,))` | `(1, 2048)` |
| 582 | `lm_to_dit_proj` | `((1, 2048),)` | `(1, 1024)` |
| 583 | `res_to_dit_proj` | `((1, 2048),)` | `(1, 1024)` |
| 584 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 585 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 586 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 587 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 588 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 589 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 590 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 591 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 592 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 593 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 594 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 595 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 596 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 597 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 598 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 599 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 600 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 601 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 602 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 603 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 604 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 605 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 606 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 607 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 608 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 609 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 610 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 611 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 612 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 613 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 614 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 615 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 616 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 617 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 618 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 619 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 620 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 621 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 622 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 623 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 624 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 625 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 626 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 627 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 628 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 629 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 630 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 631 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 632 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 633 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 634 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 635 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 636 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 637 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 638 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 639 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 640 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 641 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 642 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 643 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 644 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 645 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 646 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 647 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 648 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 649 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 650 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 651 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 652 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 653 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 654 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 655 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 656 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 657 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 658 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 659 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 660 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 661 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 662 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 663 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 664 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 665 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 666 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 667 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 668 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 669 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 670 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 671 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 672 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 673 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 674 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 675 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 676 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 677 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 678 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 679 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 680 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 681 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 682 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 683 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 684 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 685 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 686 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 687 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 688 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 689 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 690 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 691 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 692 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 693 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 694 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 695 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 696 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 697 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 698 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 699 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 700 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 701 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 702 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 703 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 704 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 705 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 706 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 707 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 708 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 709 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 710 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 711 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 712 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 713 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 714 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 715 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 716 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 717 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 718 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 719 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 720 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 721 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 722 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 723 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 724 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 725 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 726 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 727 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 728 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 729 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 730 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 731 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 732 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 733 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 734 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 735 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 736 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 737 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 738 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 739 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 740 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 741 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 742 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 743 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 744 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 745 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 746 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 747 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 748 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 749 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 750 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 751 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 752 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 753 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 754 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 755 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 756 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 757 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 758 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 759 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 760 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 761 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 762 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 763 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 764 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 765 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 766 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 767 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 768 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 769 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 770 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 771 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 772 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 773 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 774 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 775 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 776 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 777 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 778 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 779 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 780 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 781 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 782 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 783 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 784 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 785 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 786 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 787 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 788 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 789 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 790 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 791 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 792 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 793 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 794 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 795 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 796 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 797 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 798 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 799 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 800 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 801 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 802 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 803 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 804 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 805 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 806 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 807 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 808 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 809 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 810 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 811 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 812 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 813 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 814 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 815 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 816 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 817 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 818 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 819 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 820 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 821 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 822 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 823 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 824 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 825 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 826 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 827 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 828 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 829 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 830 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 831 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 832 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 833 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 834 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 835 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 836 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 837 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 838 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 839 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 840 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 841 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 842 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 843 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 844 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 845 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 846 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 847 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 848 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 849 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 850 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 851 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 852 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 853 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 854 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 855 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 856 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 857 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 858 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 859 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 860 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 861 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 862 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 863 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 864 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 865 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 866 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 867 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 868 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 869 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 870 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 871 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 872 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 873 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 874 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 875 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 876 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 877 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 878 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 879 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 880 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 881 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 882 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 883 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 884 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 885 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 886 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 887 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 888 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 889 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 890 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 891 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 892 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 893 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 894 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 895 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 896 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 897 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 898 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 899 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 900 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 901 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 902 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 903 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 904 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 905 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 906 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 907 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 908 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 909 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 910 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 911 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 912 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 913 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 914 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 915 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 916 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 917 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 918 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 919 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 920 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 921 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 922 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 923 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 924 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 925 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 926 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 927 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 928 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 929 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 930 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 931 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 932 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 933 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 934 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 935 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 936 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 937 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 938 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 939 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 940 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 941 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 942 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 943 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 944 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 945 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 946 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 947 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 948 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 949 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 950 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 951 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 952 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 953 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 954 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 955 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 956 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 957 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 958 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 959 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 960 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 961 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 962 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 963 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 964 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 965 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 966 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 967 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 968 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 969 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 970 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 971 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 972 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 973 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 974 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 975 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 976 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 977 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 978 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 979 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 980 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 981 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 982 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 983 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 984 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 985 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 986 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 987 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 988 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 989 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 990 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 991 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 992 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 993 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 994 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 995 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 996 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 997 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 998 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 999 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1000 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1001 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1002 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1003 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1004 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1005 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1006 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1007 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1008 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1009 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1010 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1011 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1012 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1013 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1014 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1015 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1016 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1017 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1018 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1019 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1020 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1021 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1022 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1023 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1024 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1025 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1026 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1027 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1028 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1029 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1030 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1031 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1032 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1033 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1034 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1035 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1036 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1037 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1038 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1039 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1040 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1041 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1042 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1043 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1044 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1045 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1046 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1047 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1048 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1049 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1050 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1051 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1052 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1053 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1054 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1055 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1056 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1057 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1058 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1059 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1060 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1061 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 1062 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 1063 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 1064 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1065 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1066 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1067 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1068 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1069 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1070 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1071 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1072 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1073 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1074 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1075 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1076 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1077 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1078 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1079 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1080 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1081 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1082 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1083 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1084 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1085 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1086 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1087 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 1088 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1089 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1090 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1091 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1092 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1093 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1094 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1095 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1096 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1097 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1098 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1099 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1100 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1101 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1102 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1103 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1104 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1105 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1106 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1107 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1108 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1109 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1110 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1111 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1112 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1113 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1114 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1115 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1116 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1117 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1118 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1119 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1120 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1121 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1122 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1123 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1124 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1125 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1126 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1127 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1128 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1129 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1130 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1131 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1132 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1133 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1134 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1135 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1136 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1137 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1138 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1139 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1140 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1141 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1142 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1143 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1144 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1145 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1146 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1147 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1148 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1149 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1150 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1151 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1152 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1153 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1154 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1155 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1156 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1157 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1158 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1159 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1160 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1161 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1162 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1163 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1164 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1165 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1166 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1167 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1168 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1169 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1170 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1171 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1172 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1173 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1174 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1175 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1176 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1177 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1178 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1179 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1180 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1181 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1182 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1183 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1184 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1185 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1186 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1187 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1188 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1189 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1190 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1191 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1192 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1193 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1194 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1195 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1196 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1197 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1198 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1199 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1200 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1201 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1202 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1203 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1204 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1205 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1206 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1207 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1208 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1209 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1210 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1211 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1212 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1213 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1214 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1215 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1216 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1217 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1218 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1219 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1220 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1221 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 1222 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 1223 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 1224 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1225 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1226 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1227 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1228 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1229 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1230 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1231 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1232 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1233 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1234 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1235 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1236 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1237 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1238 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1239 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1240 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1241 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1242 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1243 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1244 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1245 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1246 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1247 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 1248 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1249 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1250 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1251 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1252 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1253 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1254 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1255 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1256 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1257 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1258 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1259 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1260 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1261 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1262 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1263 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1264 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1265 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1266 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1267 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1268 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1269 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1270 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1271 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1272 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1273 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1274 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1275 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1276 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1277 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1278 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1279 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1280 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1281 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1282 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1283 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1284 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1285 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1286 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1287 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1288 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1289 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1290 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1291 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1292 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1293 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1294 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1295 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1296 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1297 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1298 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1299 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1300 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1301 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1302 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1303 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1304 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1305 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1306 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1307 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1308 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1309 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1310 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1311 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1312 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1313 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1314 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1315 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1316 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1317 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1318 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1319 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1320 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1321 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1322 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1323 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1324 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1325 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1326 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1327 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1328 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1329 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1330 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1331 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1332 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1333 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1334 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1335 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1336 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1337 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1338 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1339 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1340 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1341 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1342 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1343 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1344 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1345 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1346 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1347 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1348 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1349 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1350 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1351 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1352 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1353 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1354 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1355 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1356 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1357 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1358 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1359 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1360 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1361 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1362 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1363 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1364 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1365 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1366 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1367 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1368 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1369 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1370 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1371 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1372 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1373 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1374 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1375 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1376 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1377 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1378 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1379 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1380 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1381 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 1382 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 1383 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 1384 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1385 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1386 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1387 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1388 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1389 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1390 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1391 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1392 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1393 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1394 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1395 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1396 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1397 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1398 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1399 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1400 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1401 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1402 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1403 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1404 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1405 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1406 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1407 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 1408 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1409 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1410 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1411 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1412 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1413 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1414 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1415 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1416 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1417 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1418 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1419 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1420 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1421 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1422 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1423 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1424 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1425 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1426 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1427 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1428 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1429 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1430 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1431 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1432 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1433 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1434 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1435 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1436 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1437 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1438 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1439 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1440 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1441 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1442 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1443 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1444 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1445 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1446 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1447 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1448 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1449 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1450 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1451 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1452 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1453 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1454 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1455 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1456 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1457 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1458 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1459 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1460 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1461 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1462 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1463 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1464 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1465 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1466 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1467 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1468 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1469 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1470 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1471 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1472 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1473 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1474 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1475 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1476 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1477 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1478 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1479 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1480 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1481 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1482 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1483 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1484 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1485 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1486 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1487 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1488 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1489 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1490 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1491 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1492 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1493 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1494 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1495 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1496 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1497 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1498 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1499 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1500 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1501 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1502 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1503 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1504 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1505 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1506 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1507 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1508 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1509 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1510 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1511 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1512 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1513 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1514 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1515 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1516 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1517 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1518 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1519 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1520 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1521 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1522 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1523 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1524 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1525 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1526 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1527 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1528 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1529 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1530 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1531 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1532 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1533 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1534 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1535 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1536 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1537 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1538 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1539 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1540 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1541 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 1542 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 1543 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 1544 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1545 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1546 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1547 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1548 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1549 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1550 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1551 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1552 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1553 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1554 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1555 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1556 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1557 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1558 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1559 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1560 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1561 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1562 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1563 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1564 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1565 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1566 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1567 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 1568 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1569 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1570 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1571 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1572 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1573 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1574 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1575 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1576 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1577 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1578 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1579 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1580 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1581 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1582 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1583 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1584 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1585 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1586 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1587 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1588 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1589 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1590 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1591 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1592 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1593 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1594 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1595 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1596 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1597 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1598 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1599 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1600 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1601 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1602 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1603 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1604 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1605 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1606 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1607 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1608 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1609 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1610 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1611 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1612 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1613 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1614 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1615 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1616 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1617 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1618 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1619 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1620 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1621 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1622 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1623 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1624 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1625 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1626 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1627 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1628 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1629 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1630 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1631 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1632 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1633 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1634 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1635 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1636 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1637 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1638 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1639 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1640 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1641 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1642 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1643 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1644 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1645 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1646 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1647 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1648 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1649 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1650 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1651 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1652 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1653 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1654 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1655 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1656 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1657 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1658 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1659 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1660 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1661 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1662 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1663 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1664 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1665 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1666 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1667 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1668 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1669 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1670 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1671 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1672 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1673 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1674 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1675 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1676 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1677 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1678 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1679 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1680 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1681 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1682 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1683 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1684 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1685 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1686 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1687 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1688 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1689 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1690 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1691 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1692 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1693 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1694 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1695 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1696 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1697 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1698 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1699 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1700 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1701 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 1702 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 1703 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 1704 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1705 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1706 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1707 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1708 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1709 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1710 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1711 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1712 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1713 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1714 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1715 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1716 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1717 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1718 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1719 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1720 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1721 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1722 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1723 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1724 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1725 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1726 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1727 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 1728 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1729 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1730 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1731 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1732 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1733 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1734 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1735 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1736 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1737 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1738 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1739 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1740 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1741 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1742 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1743 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1744 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1745 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1746 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1747 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1748 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1749 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1750 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1751 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1752 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1753 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1754 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1755 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1756 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1757 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1758 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1759 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1760 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1761 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1762 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1763 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1764 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1765 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1766 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1767 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1768 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1769 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1770 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1771 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1772 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1773 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1774 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1775 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1776 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1777 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1778 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1779 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1780 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1781 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1782 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1783 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1784 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1785 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1786 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1787 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1788 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1789 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1790 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1791 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1792 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1793 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1794 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1795 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1796 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1797 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1798 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1799 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1800 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1801 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1802 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1803 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1804 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1805 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1806 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1807 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1808 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1809 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1810 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1811 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1812 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1813 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1814 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1815 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1816 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1817 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1818 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1819 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1820 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1821 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1822 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1823 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1824 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1825 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1826 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1827 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1828 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1829 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1830 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1831 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1832 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1833 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1834 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1835 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1836 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1837 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1838 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1839 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1840 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1841 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1842 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1843 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1844 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1845 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1846 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1847 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1848 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1849 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1850 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1851 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1852 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1853 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1854 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1855 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1856 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1857 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1858 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1859 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1860 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1861 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 1862 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 1863 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 1864 | `feat_decoder.estimator.in_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1865 | `feat_decoder.estimator.cond_proj` | `((2, 4, 64),)` | `(2, 4, 1024)` |
| 1866 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1867 | `feat_decoder.estimator.time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1868 | `feat_decoder.estimator.time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1869 | `feat_decoder.estimator.time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1870 | `feat_decoder.estimator.time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1871 | `feat_decoder.estimator.time_embeddings` | `((2,),)` | `(2, 1024)` |
| 1872 | `feat_decoder.estimator.delta_time_mlp.linear_1` | `((2, 1024),)` | `(2, 1024)` |
| 1873 | `feat_decoder.estimator.delta_time_mlp.act` | `((2, 1024),)` | `(2, 1024)` |
| 1874 | `feat_decoder.estimator.delta_time_mlp.linear_2` | `((2, 1024),)` | `(2, 1024)` |
| 1875 | `feat_decoder.estimator.delta_time_mlp` | `((2, 1024),)` | `(2, 1024)` |
| 1876 | `feat_decoder.estimator.decoder.layers.0.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1877 | `feat_decoder.estimator.decoder.layers.0.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1878 | `feat_decoder.estimator.decoder.layers.0.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1879 | `feat_decoder.estimator.decoder.layers.0.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1880 | `feat_decoder.estimator.decoder.layers.0.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1881 | `feat_decoder.estimator.decoder.layers.0.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1882 | `feat_decoder.estimator.decoder.layers.0.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1883 | `feat_decoder.estimator.decoder.layers.0.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1884 | `feat_decoder.estimator.decoder.layers.0.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1885 | `feat_decoder.estimator.decoder.layers.0.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1886 | `feat_decoder.estimator.decoder.layers.0.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1887 | `feat_decoder.estimator.decoder.layers.0` | `((11,), (2, 11, 1024), 'NoneType')` | `((2, 11, 1024), (2, 11, 1024))` |
| 1888 | `feat_decoder.estimator.decoder.layers.1.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1889 | `feat_decoder.estimator.decoder.layers.1.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1890 | `feat_decoder.estimator.decoder.layers.1.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1891 | `feat_decoder.estimator.decoder.layers.1.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1892 | `feat_decoder.estimator.decoder.layers.1.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1893 | `feat_decoder.estimator.decoder.layers.1.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1894 | `feat_decoder.estimator.decoder.layers.1.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1895 | `feat_decoder.estimator.decoder.layers.1.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1896 | `feat_decoder.estimator.decoder.layers.1.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1897 | `feat_decoder.estimator.decoder.layers.1.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1898 | `feat_decoder.estimator.decoder.layers.1.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1899 | `feat_decoder.estimator.decoder.layers.1` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1900 | `feat_decoder.estimator.decoder.layers.2.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1901 | `feat_decoder.estimator.decoder.layers.2.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1902 | `feat_decoder.estimator.decoder.layers.2.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1903 | `feat_decoder.estimator.decoder.layers.2.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1904 | `feat_decoder.estimator.decoder.layers.2.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1905 | `feat_decoder.estimator.decoder.layers.2.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1906 | `feat_decoder.estimator.decoder.layers.2.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1907 | `feat_decoder.estimator.decoder.layers.2.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1908 | `feat_decoder.estimator.decoder.layers.2.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1909 | `feat_decoder.estimator.decoder.layers.2.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1910 | `feat_decoder.estimator.decoder.layers.2.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1911 | `feat_decoder.estimator.decoder.layers.2` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1912 | `feat_decoder.estimator.decoder.layers.3.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1913 | `feat_decoder.estimator.decoder.layers.3.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1914 | `feat_decoder.estimator.decoder.layers.3.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1915 | `feat_decoder.estimator.decoder.layers.3.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1916 | `feat_decoder.estimator.decoder.layers.3.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1917 | `feat_decoder.estimator.decoder.layers.3.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1918 | `feat_decoder.estimator.decoder.layers.3.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1919 | `feat_decoder.estimator.decoder.layers.3.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1920 | `feat_decoder.estimator.decoder.layers.3.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1921 | `feat_decoder.estimator.decoder.layers.3.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1922 | `feat_decoder.estimator.decoder.layers.3.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1923 | `feat_decoder.estimator.decoder.layers.3` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1924 | `feat_decoder.estimator.decoder.layers.4.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1925 | `feat_decoder.estimator.decoder.layers.4.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1926 | `feat_decoder.estimator.decoder.layers.4.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1927 | `feat_decoder.estimator.decoder.layers.4.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1928 | `feat_decoder.estimator.decoder.layers.4.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1929 | `feat_decoder.estimator.decoder.layers.4.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1930 | `feat_decoder.estimator.decoder.layers.4.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1931 | `feat_decoder.estimator.decoder.layers.4.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1932 | `feat_decoder.estimator.decoder.layers.4.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1933 | `feat_decoder.estimator.decoder.layers.4.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1934 | `feat_decoder.estimator.decoder.layers.4.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1935 | `feat_decoder.estimator.decoder.layers.4` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1936 | `feat_decoder.estimator.decoder.layers.5.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1937 | `feat_decoder.estimator.decoder.layers.5.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1938 | `feat_decoder.estimator.decoder.layers.5.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1939 | `feat_decoder.estimator.decoder.layers.5.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1940 | `feat_decoder.estimator.decoder.layers.5.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1941 | `feat_decoder.estimator.decoder.layers.5.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1942 | `feat_decoder.estimator.decoder.layers.5.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1943 | `feat_decoder.estimator.decoder.layers.5.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1944 | `feat_decoder.estimator.decoder.layers.5.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1945 | `feat_decoder.estimator.decoder.layers.5.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1946 | `feat_decoder.estimator.decoder.layers.5.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1947 | `feat_decoder.estimator.decoder.layers.5` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1948 | `feat_decoder.estimator.decoder.layers.6.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1949 | `feat_decoder.estimator.decoder.layers.6.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1950 | `feat_decoder.estimator.decoder.layers.6.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1951 | `feat_decoder.estimator.decoder.layers.6.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1952 | `feat_decoder.estimator.decoder.layers.6.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1953 | `feat_decoder.estimator.decoder.layers.6.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1954 | `feat_decoder.estimator.decoder.layers.6.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1955 | `feat_decoder.estimator.decoder.layers.6.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1956 | `feat_decoder.estimator.decoder.layers.6.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1957 | `feat_decoder.estimator.decoder.layers.6.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1958 | `feat_decoder.estimator.decoder.layers.6.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1959 | `feat_decoder.estimator.decoder.layers.6` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1960 | `feat_decoder.estimator.decoder.layers.7.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1961 | `feat_decoder.estimator.decoder.layers.7.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1962 | `feat_decoder.estimator.decoder.layers.7.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1963 | `feat_decoder.estimator.decoder.layers.7.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1964 | `feat_decoder.estimator.decoder.layers.7.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1965 | `feat_decoder.estimator.decoder.layers.7.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1966 | `feat_decoder.estimator.decoder.layers.7.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1967 | `feat_decoder.estimator.decoder.layers.7.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1968 | `feat_decoder.estimator.decoder.layers.7.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1969 | `feat_decoder.estimator.decoder.layers.7.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1970 | `feat_decoder.estimator.decoder.layers.7.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1971 | `feat_decoder.estimator.decoder.layers.7` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1972 | `feat_decoder.estimator.decoder.layers.8.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1973 | `feat_decoder.estimator.decoder.layers.8.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1974 | `feat_decoder.estimator.decoder.layers.8.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1975 | `feat_decoder.estimator.decoder.layers.8.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1976 | `feat_decoder.estimator.decoder.layers.8.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1977 | `feat_decoder.estimator.decoder.layers.8.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1978 | `feat_decoder.estimator.decoder.layers.8.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1979 | `feat_decoder.estimator.decoder.layers.8.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1980 | `feat_decoder.estimator.decoder.layers.8.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1981 | `feat_decoder.estimator.decoder.layers.8.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1982 | `feat_decoder.estimator.decoder.layers.8.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1983 | `feat_decoder.estimator.decoder.layers.8` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1984 | `feat_decoder.estimator.decoder.layers.9.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1985 | `feat_decoder.estimator.decoder.layers.9.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1986 | `feat_decoder.estimator.decoder.layers.9.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1987 | `feat_decoder.estimator.decoder.layers.9.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 1988 | `feat_decoder.estimator.decoder.layers.9.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 1989 | `feat_decoder.estimator.decoder.layers.9.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 1990 | `feat_decoder.estimator.decoder.layers.9.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1991 | `feat_decoder.estimator.decoder.layers.9.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 1992 | `feat_decoder.estimator.decoder.layers.9.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 1993 | `feat_decoder.estimator.decoder.layers.9.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 1994 | `feat_decoder.estimator.decoder.layers.9.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1995 | `feat_decoder.estimator.decoder.layers.9` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 1996 | `feat_decoder.estimator.decoder.layers.10.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 1997 | `feat_decoder.estimator.decoder.layers.10.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 1998 | `feat_decoder.estimator.decoder.layers.10.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 1999 | `feat_decoder.estimator.decoder.layers.10.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 2000 | `feat_decoder.estimator.decoder.layers.10.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 2001 | `feat_decoder.estimator.decoder.layers.10.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 2002 | `feat_decoder.estimator.decoder.layers.10.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 2003 | `feat_decoder.estimator.decoder.layers.10.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 2004 | `feat_decoder.estimator.decoder.layers.10.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 2005 | `feat_decoder.estimator.decoder.layers.10.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 2006 | `feat_decoder.estimator.decoder.layers.10.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 2007 | `feat_decoder.estimator.decoder.layers.10` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 2008 | `feat_decoder.estimator.decoder.layers.11.input_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 2009 | `feat_decoder.estimator.decoder.layers.11.self_attn.qkv_proj` | `((2, 11, 1024),)` | `(2, 11, 2560)` |
| 2010 | `feat_decoder.estimator.decoder.layers.11.self_attn.rotary_emb` | `((22,), (2, 11, 2048), (2, 11, 256))` | `((2, 11, 2048), (2, 11, 256))` |
| 2011 | `feat_decoder.estimator.decoder.layers.11.self_attn.attn` | `((2, 11, 16, 128), (2, 11, 2, 128), (2, 11, 2, 128))` | `(2, 11, 16, 128)` |
| 2012 | `feat_decoder.estimator.decoder.layers.11.self_attn.o_proj` | `((2, 11, 2048),)` | `(2, 11, 1024)` |
| 2013 | `feat_decoder.estimator.decoder.layers.11.self_attn` | `((11,), (2, 11, 1024))` | `(2, 11, 1024)` |
| 2014 | `feat_decoder.estimator.decoder.layers.11.post_attention_layernorm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 2015 | `feat_decoder.estimator.decoder.layers.11.mlp.gate_up_proj` | `((2, 11, 1024),)` | `(2, 11, 8192)` |
| 2016 | `feat_decoder.estimator.decoder.layers.11.mlp.act_fn` | `((2, 11, 8192),)` | `(2, 11, 4096)` |
| 2017 | `feat_decoder.estimator.decoder.layers.11.mlp.down_proj` | `((2, 11, 4096),)` | `(2, 11, 1024)` |
| 2018 | `feat_decoder.estimator.decoder.layers.11.mlp` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 2019 | `feat_decoder.estimator.decoder.layers.11` | `((11,), (2, 11, 1024), (2, 11, 1024))` | `((2, 11, 1024), (2, 11, 1024))` |
| 2020 | `feat_decoder.estimator.decoder.norm` | `((2, 11, 1024),)` | `(2, 11, 1024)` |
| 2021 | `feat_decoder.estimator.decoder` | `((2, 11, 1024), (11,))` | `(2, 11, 1024)` |
| 2022 | `feat_decoder.estimator.out_proj` | `((2, 4, 1024),)` | `(2, 4, 64)` |
| 2023 | `feat_decoder.estimator` | `((2, 64, 4), (2, 2048), (2,), (2, 64, 4), (2,))` | `(2, 64, 4)` |
| 2024 | `feat_decoder` | `()` | `(1, 64, 4)` |
| 2025 | `stop_proj` | `((1, 2048),)` | `(1, 2048)` |
| 2026 | `stop_actn` | `((1, 2048),)` | `(1, 2048)` |
| 2027 | `stop_head` | `((1, 2048),)` | `(1, 2)` |

## Per-section wall time (averaged across decode steps)

- steps measured: `10`
- end-to-end wall: `180.17 ms/step`

| section | ms/step | % |
|---|---|---|
| `feat_decoder` | `124.035` | `73.5%` |
| `base_lm` | `26.397` | `15.6%` |
| `feat_encoder` | `12.775` | `7.6%` |
| `residual_lm` | `5.041` | `3.0%` |
| `fsq_layer` | `0.188` | `0.1%` |
| `enc_to_lm_proj` | `0.078` | `0.0%` |
| `lm_to_dit_proj` | `0.078` | `0.0%` |
| `res_to_dit_proj` | `0.067` | `0.0%` |
| `stop_head` | `0.055` | `0.0%` |

## Notes

- Timings measured with `enforce_eager=True` (no CUDA Graph replay). Real hot path will be faster on the no-LoRA graph.
- Section totals won't add up to 100% because we haven't instrumented every small op (projections, adds, norms outside the sections above).
