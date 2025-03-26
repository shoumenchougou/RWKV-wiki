# RWKV pip Usage Guide  

The following will guide you on **how to develop applications based on the RWKV model using the [RWKV pip library](https://pypi.org/project/rwkv/)**.  

The original code for the RWKV pip library can be found in the **[ChatRWKV](https://github.com/BlinkDL/ChatRWKV)** repository.  

## Explanation of `API_DEMO_CHAT.py`  

::: tip  
**[API_DEMO_CHAT](https://github.com/BlinkDL/ChatRWKV/blob/main/API_DEMO_CHAT.py)** is a development demo based on the RWKV pip library, designed to **implement a command-line chatbot**.  

Below, we will explain the code design of this chatbot demo in detail, with annotations for each segment.  
:::  

```python  
########################################################################################################  
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM  
########################################################################################################  

print("RWKV Chat Simple Demo")  # Print a simple message indicating this is a basic RWKV chat demo.  
import os, copy, types, gc, sys, re  # Import OS, object copying, types, garbage collection, system, regex, etc.  
import numpy as np  # Import the NumPy library  
from prompt_toolkit import prompt  # Import prompt from prompt_toolkit for command-line input  
import torch  # Import the PyTorch library  
```  

This section imports the necessary packages for inference with the RWKV model. Note the following:  

- Torch version must be at least 1.13, with 2.x+cu121 recommended.  
- You must first run `pip install rwkv`.  

---

```python  
# Optimize PyTorch settings to allow tf32  
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.allow_tf32 = True  
torch.backends.cuda.matmul.allow_tf32 = True  

# os.environ["RWKV_V7_ON"] = '1'  # Enable RWKV-7 model  
os.environ["RWKV_JIT_ON"] = "1"  # Enable JIT compilation  
os.environ["RWKV_CUDA_ON"] = "0"  # Disable native CUDA kernels. Change to '1' to enable CUDA kernels (faster but requires a C++ compiler and CUDA libraries).  
```  

::: warning  
When inferencing the RWKV-7 model, ensure `os.environ["RWKV_V7_ON"]` is set to `1`.  
:::  

These are Torch settings and environment optimizations to speed up inference.  

---

```python  
from rwkv.model import RWKV  # Import the RWKV class from the RWKV model library for loading and operating the RWKV model.  
from rwkv.utils import PIPELINE  # Import PIPELINE from the RWKV utility library for encoding/decoding data.  

args = types.SimpleNamespace()  

args.strategy = "cuda fp16"  # Model inference device and precision: CUDA (GPU) with FP16 precision.  
args.MODEL_NAME = "E://RWKV-Runner//models//rwkv-final-v6-2.1-1b6"  # Specify the path to the RWKV model (absolute path recommended).  
```  

This section imports two utility classes, `RWKV` and `PIPELINE`, from the RWKV toolkit and specifies the **device**, **precision**, and **local file path** for loading the RWKV model.  

`args.strategy` affects model generation quality and speed. ChatRWKV supports the following strategies:  

::: tip  
In the table below, `fp16i8` refers to int8 quantization on top of fp16 precision.  

Quantization reduces VRAM usage but slightly sacrifices precision compared to fp16. Therefore, if VRAM is sufficient, prefer fp16 layers.  
:::  

| Strategy | VRAM & RAM | Speed |  
|----------|------------|-------|  
| **cpu fp32** | 7B model requires 32GB RAM | Uses CPU fp32 precision, suitable for Intel. Very slow on AMD due to PyTorch's CPU gemv issues and single-core execution. |  
| **cpu bf16** | 7B model requires 16GB RAM | Uses CPU bf16 precision. Faster on newer Intel CPUs (e.g., Xeon Platinum) supporting bfloat16. |  
| **cpu fp32i8** | 7B model requires 12GB RAM | Uses CPU int8 quantization. Slower than `cpu fp32`. |  
| **cuda fp16** | 7B model requires 15GB VRAM | Loads all layers in fp16 precision. Fastest but demands the most VRAM. |  
| **cuda fp16i8** | 7B model requires 9GB VRAM | Quantizes all layers to int8. Faster if `os.environ["RWKV_CUDA_ON"] = '1'` is set to compile CUDA kernels (reduces VRAM usage by 1–2GB). |  
| **cuda fp16i8 \*20 -> cuda fp16** | VRAM usage between fp16 and fp16i8 | Quantizes the first 20 layers (`*20` denotes layer count) to fp16i8, loading the rest in fp16. Adjust the number of fp16i8 layers based on remaining VRAM. |  
| **cuda fp16i8 \*20+** | Uses less VRAM than fp16i8 | Quantizes the first 20 layers to fp16i8 and fixes them on the GPU, dynamically loading other layers (3x slower but saves VRAM). Reduce fixed layers if VRAM is insufficient. |  
| **cuda fp16i8 \*20 -> cpu fp32** | Uses less VRAM than fp16i8 but more RAM | Quantizes the first 20 layers to fp16i8 on GPU and loads the rest on CPU in fp32. Faster if CPU performance is strong. Adjust GPU layers based on VRAM availability. |  
| **cuda:0 fp16 \*20 -> cuda:1 fp16** | Uses dual GPUs | Loads the first 20 layers on GPU 0 (cuda:0) in fp16 and the rest on GPU 1 (cuda:1). Use fp16i8 if VRAM is insufficient on either GPU. |  

---

```python  
# STATE_NAME = None  # Do not use State  

# Specify the path to the State file.  
STATE_NAME = "E://RWKV-Runner//models//rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048"  # Absolute path to the custom State file.  
```  

This section determines whether to load a State file. `"None"` means no custom State is loaded. Specify an absolute path to load a State file.  

::: tip  
State is a unique feature of RNN-like models such as RWKV. By loading custom State files, the model's performance in specific tasks can be enhanced (similar to a plugin).  
:::  

---

```python  
# Set model decoding parameters  
GEN_TEMP = 1.0  
GEN_TOP_P = 0.3  
GEN_alpha_presence = 0.5  
GEN_alpha_frequency = 0.5  
GEN_penalty_decay = 0.996  

# If a State file is loaded, adjust generation parameters for better responses.  
if STATE_NAME != None:  
    GEN_TOP_P = 0.2  
    GEN_alpha_presence = 0.3  
    GEN_alpha_frequency = 0.3  

CHUNK_LEN = 256  # Chunk input text for processing  
```  

This section configures **decoding parameters** for the RWKV model, with different settings depending on whether a State file is loaded.  

::: tip  
For details on RWKV decoding parameters, refer to the [RWKV Decoding Parameters Documentation](../basic/decoding-parameters).  
:::  

When a custom State file is loaded, we adjust the parameters to better align with the State's format and style, **lowering the `top_p` and penalty parameters**.  

`CHUNK_LEN` splits input text into chunks of the specified size. Larger values allow **more parallel text processing** but **consume more VRAM**. Reduce to 128 or 64 if VRAM is limited.  

---

```python  
print(f"Loading model - {args.MODEL_NAME}")  # Print model loading message  
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)  # Load the RWKV model.  
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")  # Initialize PIPELINE with the RWKV-World vocabulary for encoding/decoding.  
```  

This section loads the RWKV model using the previously set **strategy** and **decoding parameters**.  

To add a confirmation message after loading, insert: `print(f"{args.MODEL_NAME} - Model loaded successfully")` at the end.  

---

```python  
model_tokens = []  
model_state = None  

# If STATE_NAME is specified, load the custom State file and initialize the model State.  
if STATE_NAME != None:  
    args = model.args  # Get model parameters  
    state_raw = torch.load(STATE_NAME + '.pth')  # Load State data from the specified file  
    state_init = [None for i in range(args.n_layer * 3)]  # Initialize State list  
    for i in range(args.n_layer):  # Loop through each layer  
        dd = model.strategy[i]  # Get the loading strategy for each layer  
        dev = dd.device  # Get the device (e.g., GPU) for each layer  
        atype = dd.atype  # Get the data type (FP32/FP16 or int8, etc.)  
        # Initialize the model State  
        state_init[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()  
        state_init[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()  
        state_init[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()  
    model_state = copy.deepcopy(state_init)  # Copy the initialized State  
```  

This code loads a custom State file and writes it into the model's initial State.  

This section usually requires no modifications.  

---

```python  
def run_rnn(ctx):  
    # Define two global variables to update tokens and model State  
    global model_tokens, model_state  
    ctx = ctx.replace("\r\n", "\n")  # Convert CRLF (Windows line breaks) to LF (Linux line breaks)  
    tokens = pipeline.encode(ctx)  # Encode text into tokens using the RWKV model's vocabulary  
    tokens = [int(x) for x in tokens]  # Convert tokens to integers for consistency  
    model_tokens += tokens  # Append tokens to the global model token list  

    while len(tokens) > 0:  # Process tokens in chunks until all are handled  
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)  # Forward pass, processing CHUNK_LEN tokens and updating State  
        tokens = tokens[CHUNK_LEN:]  # Remove processed tokens and continue with the rest  

    return out  # Return the model's prefill result  
```  

This function controls the RWKV model's prefill in RNN mode. It splits `ctx` (context) into segments of length `CHUNK_LEN`, processes them sequentially in the RNN, and returns the final `model_state` and `out`.  

The function takes a `ctx` parameter (usually a **string**). It then processes the text and tokens as follows:  

1. Uses `replace` to standardize line breaks to `\n`, as RWKV's training dataset uses `\n` as the standard line break.  
2. Uses `pipeline.encode` to convert input text into tokens based on the RWKV-World vocabulary.  
3. Converts tokens to integers for consistency.  
4. Performs forward propagation on the tokens, updating the model State and returning `out`.  

::: warning  
Note: The function returns `out`, which is not an actual token or text but the model's raw prediction (tensor) for the next token.  
:::  

To convert `out` into actual tokens or text, sampling (e.g., via `pipeline.sample_logits`) is required to predict the next **token**, which is then decoded into **text**.  

---

```python  
# If no custom State is loaded, use an initial prompt for the conversation  
if STATE_NAME == None:  
    init_ctx = "User: hi" + "\n\n"  
    init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"  
    run_rnn(init_ctx)  # Run RNN prefill on the initial prompt  
    print(init_ctx, end="")  # Print the initial dialogue text  
```  

If no State file is loaded, a **default dialogue prompt** is used for prefill.  

---

```python  
# Read user input, loop to generate the next token  
while True:  
    msg = prompt("User: ")  # Read user input into msg  
    msg = msg.strip()  # Remove leading/trailing whitespace  
    msg = re.sub(r"\n+", "\n", msg)  # Replace multiple line breaks with a single one  
    if len(msg) > 0:  # If the processed input is non-empty  
        occurrence = {}  # Dictionary to track token occurrences for repetition penalty  
        out_tokens = []  # List to store output tokens  
        out_last = 0  # Tracks the last generated token position  

        out = run_rnn("User: " + msg + "\n\nAssistant:")  # Prefill with user input formatted as RWKV dialogue  
        print("\nAssistant:", end="")  # Print "Assistant:" label  

        for i in range(99999):  
            for n in occurrence:  
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency  # Apply presence and frequency penalties  
            out[0] -= 1e10  # Disable END_OF_TEXT  

            token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)  # Sample the next token  

            out, model_state = model.forward([token], model_state)  # Forward pass  
            model_tokens += [token]  
            out_tokens += [token]  # Add the new token to the output list  

            for xxx in occurrence:  
                occurrence[xxx] *= GEN_penalty_decay  # Apply penalty decay  
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)  # Update token occurrence count  

            tmp = pipeline.decode(out_tokens[out_last:])  # Decode the latest tokens into text  
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # If the text is valid UTF-8 and doesn't end with a line break  
                print(tmp, end="", flush=True)  # Print the decoded text in real-time  
                out_last = i + 1  # Update the output position  

            if "\n\n" in tmp:  # If the text contains a double line break (can be replaced with other stop tokens)  
                print(tmp, end="", flush=True)  # Print the decoded text  
                break  # End inference  
    else:  
        print("!!! Error: please say something !!!")  # Prompt if user input is empty  
```  

**This section implements the core loop for receiving user input, performing RNN-mode inference, and generating text.**  

The main logic is as follows:  

1. **Process user input**: Normalize spaces and line breaks, check input length.  
   - If empty after normalization, prompt the user to "say something."  
   - If non-empty, proceed to step 2.  
2. **Prefill**: Format the input into a dialogue prompt and prefill to obtain logits.  
3. **Generate tokens and print text**:  
   - Apply presence (`GEN_alpha_presence`) and frequency (`GEN_alpha_frequency`) penalties.  
   - Sample the next token using `temperature` and `top_p` parameters.  
   - Perform forward propagation with the new token for the next prediction.  
   - Adjust token probabilities using penalty decay (`penalty_decay`).  
   - Decode the generated tokens into text.  
   - Print the text in real-time and check for stop tokens (`\n\n`). If found, exit inference.  

---

From the inference process, we can see that the model updates its hidden state (State) at each timestep and uses it to generate the next output. This aligns with the core RNN behavior: **the model's output depends on the previous generation step**.  

---
