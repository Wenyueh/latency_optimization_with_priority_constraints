import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer 
from tqdm import tqdm
import argparse, json, os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def compute_time(model, tokenizer, input_multiple, decode_length):
    if input_multiple == 0:
        original_prompt = "Test "
        filler = original_prompt*input_multiple*10
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": original_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt = filler + text
        prompt = prompt[:-1] # remove the last newline character
        model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda:0')
    else:
        original_prompt = "Please provide me a short introduction to large language model. "
        filler = original_prompt*(input_multiple-1)*10
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": original_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt = filler + text
        prompt = prompt[:-1] # remove the last newline character
        model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda:0')

    input_length = list(model_inputs['input_ids'].size())[1]
    start_time = time.time()
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=decode_length
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    end_time = time.time()
    running_time = end_time-start_time
    return input_length, decode_length, running_time


def prefill_function(x, alpha, beta):
    return alpha * x ** 2 + beta * x

def decoding_function(X, alpha, beta):
    x1 = X[0]
    x2 = X[1]
    return alpha * (1/2 * x2 ** 2 + x1 * x2 + 1/2 * x2) + beta * x2

def collect_prefill_data(files):
    # prefill {k1:v1, k2:v2, ...}
    # f(k) = alpha * k^2
    files = [f for f in files if 'prefill' in f]
    prefill = {}
    for f in files:
        with open(os.path.join(args.data_dir, f)) as file:
            data = json.load(file)
        for prefill_length, prefill_time in data.items():
            if prefill_length not in prefill:
                prefill[prefill_length] = []
            prefill[prefill_length].append(prefill_time)

    prefill = {k: np.mean(v) for k, v in prefill.items()}
    if '20' in prefill:
        prefill.pop('20')

    return prefill 

def collect_decoding_data(files):
    # decoding {(k1, k1'): v1, (k2, k2'): v2, ...}
    # f(k, k') = beta * (1/2 * k'^2 + k * k' + 1/2 * k')
    files = [f for f in files if 'decoding' in f]
    decoding = {}
    for f in files:
        with open(os.path.join(args.data_dir, f)) as file:
            data = json.load(file)
        for prefill_length, decoding_result in data.items():
            for decoding_length, decoding_time in decoding_result.items():
                if (prefill_length, decoding_length) not in decoding:
                    decoding[(prefill_length, decoding_length)] = []
                decoding[(prefill_length, decoding_length)].append(decoding_time)

    decoding = {k: np.mean(v) for k, v in decoding.items()}
    
    return decoding 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='profile')
    parser.add_argument('--GPU', type=str, default='L40s')
    args = parser.parse_args()

    torch.random.manual_seed(0) 
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-4B-Chat",
        torch_dtype="auto",
        device_map='cuda:0'
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")
    # original 10 tokens

    ########## save
    prefill_time = {}
    decoding_time = {}
    for i in range(51):
        input_length, decode_length, running_time = compute_time(model, tokenizer, i, decode_length=1)
        if i != 0:
            prefill_time[input_length] = running_time
            decoding_time[input_length] = {}
            for j in tqdm(range(1, 1001, 5), desc=f"input_length={input_length}"):
                input_length, decode_length, running_time = compute_time(model, tokenizer, i, decode_length=j)
                decoding_time[input_length][decode_length] = running_time
                torch.cuda.empty_cache()

            with open('profile_GPU/L40s_prefill_time.json', 'w') as f:
                json.dump(prefill_time, f)

            with open('profile_GPU/L40s_decoding_time.json', 'w') as f:
                json.dump(decoding_time, f)

    ######### compute
    files = os.listdir(args.data_dir)
    files = [f for f in files if args.GPU in f]

    prefill = prefill_function_result = collect_prefill_data(files)
    decoding = decoding_function_result = collect_decoding_data(files)

    prefill_xdata = list(prefill.keys())
    prefill_ydata = list(prefill.values())

    # Fit the curve
    prefill_speed, _ = curve_fit(prefill_function, prefill_xdata, prefill_ydata)
    print(prefill_speed)
    #plot the original data and the curve fitting
    # plt.plot(prefill_xdata, prefill_ydata, label='original data')
    # plt.plot(prefill_xdata, [prefill_function(float(x), prefill_speed[0], prefill_speed[1]) for x in prefill_xdata], label='curve fitting')
    # plt.show()

    decoding_xdata1 = [int(k[0]) for k in decoding.keys()]
    decoding_xdata2 = [int(k[1]) for k in decoding.keys()]
    decoding_ydata = list(decoding.values())
    # Fit the curve
    decoding_speed, _ = curve_fit(decoding_function, (decoding_xdata1, decoding_xdata2), decoding_ydata)
    print(decoding_speed)