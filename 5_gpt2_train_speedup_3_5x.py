# speedup stat:
# torch.autocast                                    2.3x, 1.05x
# F.scaled_dot_product_attention(flash attention)   1.4x
# fine number 50257 -> 50304                        1.04x
# total speedup: 2.3 * 1.05 * 1.4 * 1.04 = 3.5x

# unused speedup:
# torch.set_float32_matmul_precision(1.5x). this speedup has no effect when using torch.autocast
# torch.compile(2.3x). 3080 has no effect. speedup only V100, A100, or H100. 
# total speedup not used: 1.5 * 2.3 = 3.4x
import torch, math, tiktoken, time
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
from torch.nn import functional as F


# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print('using device', device)


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask but following the OpenAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2(124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T ,hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # 1.4x faster. flash attension. fused version of the upper 4 lines

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    def _init_weights(self, module):
        std_default = 0.02  # gpt2 used!!!!!!!!!!!!!: 
        # std_default = 1 / math.sqrt(self.config.n_embd)  # near 0.036. gpt2 suggest: 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std_default *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std_default)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std_default)

    # @torch.autocast(device_type=device, dtype=torch.bfloat16)
    # @torch.autocast(device_type=device)  # speedup only forward and loss, no backward and optim. 2.3x faster in 3080 16G laptop. 1.05x faster than using param 'dtype=torch.bfloat16'
    # @torch.compile() # speedup only V100, A100, or H100. https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    def forward(self, x):
        # x is of shape (B, T)
        B, T = x.size()
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(x) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


class DataloaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        # tiny shakespeare dataset
        # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        print(f'loaded {len(tokens)} tokens.')
        print(f'1 epoch = {len(tokens) // (B * T)} batches.')
        self.tokens = torch.tensor(tokens).to(device)
        # self.tokens = torch.tensor(tokens)

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[ : -1]).view(B, T) # inputs
        y = (buf[1 : ]).view(B, T)  # target
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

data_loader = DataloaderLite(B = 16, T = 512)

# torch.set_float32_matmul_precision('high')  # 'highest'; 'high'; 'medium'. 1.5x faster than org, but torch.autocast do better

# model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig()) # default vocab_size=50257
model = GPT(GPTConfig(vocab_size=50304)) # 1.04x faster. because 50304/128 = 393
# model = GPT(GPTConfig(vocab_size=50432)) # 1.04x faster. because 50432/256 = 197
# model = GPT(GPTConfig(vocab_size=50688)) # 1.04x faster. because 50688/512 = 99
# model = GPT(GPTConfig(vocab_size=51200)) # 1.04x faster. because 51200/1024 = 50
model.to(device)
# model = torch.compile(model, mode='reduce-overhead') # speedup only V100, A100, or H100. https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
    t0 = time.time()
    x, y = data_loader.next_batch()
    # x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.autocast(device_type=device):  # speedup only forward and loss, no backward and optim. 2.3x faster in 3080 16G laptop. 1.05x faster than using param 'dtype=torch.bfloat16'
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    # import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000   # time difference in miliseconds
    tokens_per_sec = (data_loader.B * data_loader.T) / (t1 - t0)
    print(f'step {i}, loss: {loss.item():.2f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}')
