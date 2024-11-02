## Relative multihead self attention
Proposed in https://arxiv.org/pdf/1809.04281.pdf
This implementation also allows:
- KV caching, also avoids redundant relative transformations
- Unmasked attention, which involves learning both relative future and relative past transformations
- Relative attention bias

### How does relative attention work?
Similar to regular attention, but now the attention score computation involves an additional matrix Srel: Attention = Softmax((Q*Kt + Srel)/sqrt(dim)). To compute Srel, we learn max_length linear transformations for each head (max_length*2 if bidirectional). To obtain an entry i, j in Srel we apply [i-j]th linear transformation ([j-i]th if bidirectional and j>i) to Qi. Naturally, this limits maximum sequence length.

When used in combination with sinusoidal positional encoding, it allows the transformer to attend to both relative and absolute positions. In theory it should perform better than RoPE in tasks that don't require long attention windows, due to higher flexibility.

### KV cache usage example: autoregressive decoding:
```python
#in decoder layer
def __init__(self, ...):
    ...
    self.mha = RelativeGlobalAttention(d=d_model, h=num_heads, max_seq=seq_length, bidirectional=False, rel_attn_bias=False)
    ...

def forward(self, x, mask=mask, is_inference):
    ...
    mha_out, _ = self.mha(mha_in, mask=mask, cache_enabled=is_inference) 
    ...

#somewhere in the model class
def reset_kv_cache(self):
    for layer in self.decoder_layers:
        for module in layer.modules():
            if isinstance(module, RelativeGlobalAttention):
                module.kv_cache = None

def generate(self, x, desired_length):

    self.reset_kv_cache()
    ...
    #forward all context without the last token to cache it
    for i in range(desired_length):
        #forward only last token from the sequence one by one
        #sample and extend the sequence
```
