# Tiny Tales GPT

### <a href="https://kunalmishra.info"><small>Website</small></a>

<b>I trained and deployed a 30M parameter LLM trained on 1B tokens into production under $15. (Training cost $12 , DNS cost - $3).</b> <br>When we think of training LLMs the only thing that comes to our mind are those billion , trillion parameters and the exorbitant cost to train them. This project was done to show that if you make the domain narrow enough you can train your custom LLMs with smaller number of parameters that can outperform a generalized LLM. Tiny Tales GPT is 33 times smaller than GPT 2 and can still capture grammar, punctuation, diversity, and reasoning capabilities.

## Steps Involved
- Data Collection
- Data Tokenization
- Model Architecture
- Training the model
- Inference
- Putting the Web Server
- Deploying the Web Server on the Internet


### 1. Data Collection

Paper link - [link](https://arxiv.org/abs/2305.07759) <br>
Dataset link - [link](https://huggingface.co/roneneldan/TinyStories-1M)

The Dataset that is used to train the model is very unique. It is a synthetic dataset of short stories that only contain words that a typical 3 to 4-year-olds usually understand. The total number of short stories in the dataset is 1 Million.  The total size of the raw dataset is 8 GB.

### 2. Data Tokenization

I had two choices for Byte pair encoding tokenizer tiktoken or sentencepiece. I chose to go with sentencepiece as it has less vocab size (hence less paramters). 




