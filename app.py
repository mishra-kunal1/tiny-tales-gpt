from flask import Flask, render_template, request, jsonify
import os
from contextlib import nullcontext
import torch
from llama_model import ModelArgs, Transformer
from tokenizer import Tokenizer
import time

# url = " https://drive.google.com/uc?id=1vxbT0bJYPXhpSP71Xw_F5j0O-J8KPxDD"
# output = "ckpt_100k.pt"
# gdown.download(url, output)

out_dir = './' # ignored if init_from is not 'resume'
 # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
#num_samples = 2 # number of samples to draw

max_new_tokens = 200 # number of tokens generated in each sample
temperature = 0.6 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float16"
compile = False # use PyTorch 2.0 to compile the model to be faster
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
ckpt_path = os.path.join(out_dir, 'model_100k.pt')


# load the tokenizer
enc = Tokenizer()
app = Flask(__name__)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = checkpoint['model_args']
model = Transformer(ModelArgs(**gptconf))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)
# Load your machine learning model
#model = joblib.load('path/to/your/model.pkl')
# Load your machine learning model



# Load your machine learning model
#model = joblib.load('path/to/your/model.pkl')
# Load your machine learning model

@app.route('/')
def index():
    return render_template('./index.html')

def write_log(log_message):
    with open('log.txt', 'a') as log_file:  # Open the existing log file in append mode
        log_time = time.strftime('%Y-%m-%d %H:%M:%S') # Get current time
        log_file.write(f"{log_time} - {log_message}\n")

@app.route('/generate_text', methods=['POST'])
def generate_text():
    # Get input from the form
    prompt = request.form['prompt']
    start_ids = enc.encode(prompt, bos=True, eos=False)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    num_samples = int(request.form['num_samples'])  # Get the number of samples`
    max_new_tokens=int(request.form['max_tokens'])
    generated_texts = []
    start=time.time()
    incomplete=False
    with torch.no_grad():
            with ctx:
                for _ in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    decoded_tokens = y[0].tolist()
                    try:
                        clip_index = (decoded_tokens[1:]).index(1)
                    except ValueError:
                        clip_index = len(decoded_tokens)
                        incomplete=True
                    generated_text = enc.decode(decoded_tokens[:clip_index + 1])
                    if(incomplete):
                        incomplete_str='.....(max_tokens limit reached)'
                        generated_text+=incomplete_str
                    generated_texts.append(generated_text)
                    print('*' * 80)
    end=time.time()
    time_in_seconds = round(end-start,2)
    #print('Time taken:', time_in_seconds)
    write_log(f"Prompt: {prompt}, Num Samples: {num_samples} , Time taken: {time_in_seconds} seconds, Max Tokens: {max_new_tokens}")
    return render_template('result.html', generated_texts=generated_texts)


if __name__ == '__main__':
    #app.run(debug=True)
     app.run(host='127.0.0.1', port=8081, debug=True)
