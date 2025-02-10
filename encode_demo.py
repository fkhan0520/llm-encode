import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("mps")

model = model.to(device)

def pad(s):
    """
    Qwen tokenizer has vocab size of ~150k, so can be represented by 6 base 10 digits
    """
    zeros = 6 - len(s)
    return "0"*zeros + s

def process_encode(input_text, encoding_prompt):
    encoded_input_text = tokenizer.encode(input_text, return_tensors="pt").to(device)
    padded_toks = [pad(str(int(x))) for x in encoded_input_text[0]]
    expanded = [int(d) for tok in padded_toks for d in tok]

    tokenized_preamble = tokenizer.encode(encoding_prompt)
    K = len(tokenized_preamble)

    cur_prompt = tokenized_preamble
    for en in tqdm.tqdm(expanded):
        with torch.no_grad():
            if len(cur_prompt) == 0:
                inputs = None
            else:
                inputs = torch.tensor(cur_prompt, dtype=torch.int64, device=device)[None, :]
            cur_outputs = model.generate(inputs=inputs, max_new_tokens=1, output_scores=True, do_sample=False, return_dict_in_generate=True)
        cur_logits = cur_outputs.scores[0][0]
        sorted_indices = torch.argsort(cur_logits, descending=True)
        cur_prompt.append(sorted_indices[en].item())

    encoded_toks = torch.tensor(cur_prompt)
    encoded_text = tokenizer.decode(encoded_toks, skip_special_tokens=True)
    
    show_intermediate = str(tokenizer.encode(input_text)) + "\n" \
        + str(padded_toks) + "\n" + str(expanded)
    
    print(encoded_text, len(encoded_text))

    return show_intermediate, encoded_text

def process_decode(input_text, encoding_prompt):
    encoded = tokenizer.encode(input_text, return_tensors="pt")[0]
    
    tokenized_preamble = tokenizer.encode(encoding_prompt)
    K = len(tokenized_preamble)
    
    decoded = []
    print(f"{encoded=} {tokenized_preamble=} {K=} {encoded.shape=}")
    for i in tqdm.tqdm(range(K, encoded.shape[0])):
        inputs = None
        if i > 0:
            inputs = encoded[:i][None, :].to(device)
        with torch.no_grad():
            cur_outputs = model.generate(inputs=inputs, max_new_tokens=1, output_scores=True, do_sample=False, return_dict_in_generate=True)
        cur_logits = cur_outputs.scores[0][0]
        sorted_indices = torch.argsort(cur_logits, descending=True)
        rank = (sorted_indices == encoded[i]).nonzero().item()
        decoded.append(rank)
    
    strs = [str(x) for x in decoded]
    chunked = []
    idx = 0
    while idx <= len(decoded) - 6:
        print(idx, len(decoded))
        chunked.append("".join(strs[idx:idx+6]))
        idx += 6
    
    decoded_toks = [int(x) for x in chunked]
    decoded_str = tokenizer.decode(torch.tensor(decoded_toks), skip_special_tokens=True)

    show_intermediate = str(encoded.tolist()) + "\n" + str(decoded) + "\n" + str(chunked) + "\n" + str(decoded_toks)

    print(input_text, len(input_text))
    return show_intermediate, decoded_str

with gr.Blocks() as demo:
    encoding_prompt = gr.Textbox(label="Encoding Prompt")
    with gr.Tabs():
        with gr.Tab("Encode"):
            with gr.Column():
                encode_input = gr.Textbox(label="Input Text")
                encode_intermediate = gr.Textbox(label="Intermediate Result", interactive=False)
                encode_output = gr.Textbox(label="Encoded", interactive=False)
                encode_button = gr.Button("Encode")
            
            encode_button.click(
                fn=process_encode,
                inputs=[encode_input, encoding_prompt],
                outputs=[encode_intermediate, encode_output]
            )

        with gr.Tab("Decode"):
            with gr.Column():
                decode_input = gr.Textbox(label="Encoded")
                decode_intermediate = gr.Textbox(label="Intermediate Result", interactive=False)
                decode_output = gr.Textbox(label="Input Text", interactive=False)
                decode_button = gr.Button("Decode")
            
            decode_button.click(
                fn=process_decode,
                inputs=[decode_input, encoding_prompt],
                outputs=[decode_intermediate, decode_output]
            )

if __name__ == "__main__":
    demo.launch()