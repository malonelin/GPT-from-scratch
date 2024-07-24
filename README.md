# GPT-from-scratch
This is a nanoGPT, Skeleton of GPT2 with GPT3 hyper parameter optimization
Most of the code comes from https://github.com/karpathy/build-nanogpt
I am trying to let it more concise and easier to be reused.

If you want to know the model structure of GPT, read 0_gpt2_model.py. 
If you want to train the most nano GPT, use 6_gpt2_train_hyparam.py. It contains the best practices, speedup skills and the GPT3 hyper parameter.

Change B, T paramter to fix the memory size to your own CUDA device.



Content:

0_gpt2_model.py
  gpt2 skeleton model in around 100 lines
  
1_gpt2_hf_eval.py
  loading gpt2 hugging face pretrain model to eval model

  
2_gpt2_train_128words.py
  train 128 words code
  
3_gpt2_train_dataloader.py
  use tiny shakespeare dataset to train the model
  # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  
4_gpt2_train_best_practices.py
  best practices for GPT2 model
    # 1. output 'lm_head' and input embedding 'wte' use the same weight param
    # 2. init different weight using different std

5_gpt2_train_speedup_3_5x.py
  # speedup stat:
  # torch.autocast                                    2.3x, 1.05x
  # F.scaled_dot_product_attention(flash attention)   1.4x
  # fine number 50257 -> 50304                        1.04x
  # total speedup: 2.3 * 1.05 * 1.4 * 1.04 = 3.5x
  
  # unused speedup:
  # torch.set_float32_matmul_precision(1.5x). this speedup has no effect when using torch.autocast
  # torch.compile(2.3x). 3080 has no effect. speedup only V100, A100, or H100. 
  # total speedup not used: 1.5 * 2.3 = 3.4x
  
6_gpt2_train_hyparam.py
  # 1. nn.utils.clip_grad_norm_
  # 2. GPT3 125M Learning Rate 6e-4. warmup, cosine decay
  # 3. weight decay 0.1(gpt3) for matmuls and embeddings, except for biases and layernorms
  # 4. use fused parameter in AdamW, speedup 1.03x
  # 5. GPT3 125M BatchSize 0.5M, use micro batch to accummulate grad (grad_accum_steps)



ref:
GPT2 paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
GPT3 paper: https://arxiv.org/pdf/2005.14165
org src: https://github.com/karpathy/build-nanogpt
video: https://youtu.be/l8pRSuU81PU
