##########====================================================================################
##########====================PRODUCTION VERSION -- FOR GRADIO=====================###########
##########====================================================================################
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_ENABLE_XET"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "info"

from typing import List, Dict, Tuple
from datetime import datetime
from huggingface_hub import login, snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch, threading
from anthropic import Anthropic
import time, gradio as gr


HF_TOKEN = os.getenv("HF_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
assert ANTHROPIC_API_KEY, "Set ANTHROPIC_API_KEY in Space settings"

if HF_TOKEN:
    login(HF_TOKEN, add_to_git_credential=True)

QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CLAUDE_MODEL = "claude-3-5-haiku-latest"

claude = Anthropic(api_key=ANTHROPIC_API_KEY)

### Lazy loader
model = None
tokenizer = None
cache_path = "./hf-cache"
_model_lock = threading.Lock()

def enable_model() -> None:
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return

    with _model_lock:  ### prevent double snapshot/load
        if model is not None and tokenizer is not None:
            return

        os.makedirs(cache_path, exist_ok=True)
        expected_model_dir = os.path.join(cache_path, QWEN_MODEL.replace("/", "--"))

        if not os.path.exists(expected_model_dir):
            model_path = snapshot_download(
                repo_id=QWEN_MODEL,
                cache_dir=cache_path,
                local_dir=expected_model_dir,
                local_dir_use_symlinks=False,
                token=os.getenv("HF_TOKEN"),
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "model.safetensors",
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                ],
                resume_download=True,
            )
        else:
            model_path = expected_model_dir

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0},
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            quantization_config=bnb,
        ).eval()


def invoke_messages(
        rows_num: int,
        business_category: str,
        columns: str,
        instruction: str,
) -> List[Dict[str, str]]:
    system_message = """
        You are a helpful assistant generating synthetic mockup dataset as per
        user's request across all types of businesses and sorts.
        User's specific request for the data niche, data column types, and all
        other details and your job is to create wonderful mockup data for them
        to use for their demo apps or develop in a testing environment.
    """.strip()

    user_prompt = f"""
        Generate a synthetic mockup data that fits the following instruction:
        - Number of rows: {rows_num}
        - Business area: {business_category}
        - Columns: {columns}
        - Other instruction: {instruction}
        ㅡ Make sure to deliver only the markdown content without any additional comments
    """.strip()

    system_message = system_message + """
        In the case of sql file selection as an output, make sure to
        contain the full sql file format, including CREATE TABLE command.
    """.strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    return messages


def pass_claude_msg(file_format: str, content: str) -> Tuple[str, str]:
    claude_sys_msg = """
        You are a helpful assistant, converting generated outputs (done by other model)
        into the format of chosen type:
        example: csv, sql, or json format.
        NOTE: generate the result output that only includes the markdown content
        without any addtional comments!
    """.strip()
    claude_user_msg = f"""
        Convert the output into the {file_format} format for the following content:
        ----------------------------------------------------------------------
        {content}
    """.strip()

    return claude_sys_msg, claude_user_msg


def generate_output(messages):
    enable_model()

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,  ### IMPORTANT: to get a mapping
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_attention_mask=True
    ).to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2
    )

    ### Get the length(num of tokens) of the input prompt
    prompt_len = inputs["input_ids"].shape[1]

    ### Slice the generated sequence to skip the prompt length
    gen_tokens = outputs[0][prompt_len:]


    return gen_tokens


def launch_claude_api(sys_msg, user_msg):
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        system=sys_msg,
        max_tokens=400,
        temperature=0.1,
        messages=[
            {"role": "user", "content": user_msg}
        ]
    )
    return response.content[0].text


###============= Gradio Function =============###

def generate_mockup_data(category, num_data_rows, columns, a_instruction,
                         progress=gr.Progress()):
    progress(0.2, desc="Generating...")
    msg = invoke_messages(
        rows_num=int(num_data_rows or 10),
        business_category=category,
        columns=columns,
        instruction=a_instruction
    )

    res = {"toks": None, "err": None}
    def _work():
        try:
            res["toks"] = generate_output(msg)
        except Exception as e:
            res["err"] = e

    t = threading.Thread(target=_work, daemon=True)
    t.start()

    pct = 0.05
    progress(pct, desc="Generating...")
    while t.is_alive():
        ### avoid to go past 0.95
        ### if model is already loaded, the progress bar moves faster
        ### to properly simulate the loading speed
        if model:
            pct = min(0.99, pct + 0.005)
        else:
            pct = min(0.99, pct + 0.001)
        progress(pct, desc="Generating...")
        time.sleep(0.1)

    t.join()
    if res["err"]:
        raise gr.Error("Error occurred.")
    progress(1.0, desc="Done")

    return tokenizer.decode(res["toks"], skip_special_tokens=True)


def show_hidden_row():
    return gr.update(visible=True)


def make_file(btn_sort: str, category: str, content: str):
    '''
    btn_sort: one of the 3 download file tpes from the buttons -- download csv, sql, json
    category: Business category or area that the data is associated with.
    content: LLM generated text output to write in a file
    '''

    if not content or not content.strip():
        raise gr.Error("The result content is empty. Cannot create a file.")

    try:
        sys_msg, user_msg = pass_claude_msg(btn_sort, content)
        claude_output = launch_claude_api(sys_msg, user_msg)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"/tmp/{category}_mockup_{ts}.{btn_sort}"

        with open(filepath, "w") as f:
            f.write(claude_output)

        return filepath
    except Exception as e:
        raise gr.Error("Failed to format or create the file.")


###============= Gradio UI =============###

def render_interface():

    with gr.Blocks(title="Mockup Data Generator", css="footer {visibility:hidden}") as demo:
        category = gr.Textbox(
            label="Business Area/Category",
            placeholder="e.g. HR, Sales, Hospitality, Senior Care, E-commerce, Finance",
        )
        num_data_rows = gr.Number(
            label="Number of Rows",
            placeholder="Type number...",
            minimum=10,
            maximum=50,
            step=10,
            precision=0
        )
        columns = gr.Textbox(
            label="Insert Columns",
            placeholder="Comma, separated..."
        )
        a_instruction = gr.Textbox(
            label="Additional Instruction",
            placeholder="Any additional instruction. Leave blank if none.",
            lines=5
        )
        btn = gr.Button(
            value="Generate"
        )
        out = gr.Textbox(label="Result shown here.")

        buttons_row = gr.Row(visible=False)

        with buttons_row:
            btn_csv = gr.DownloadButton(label="Download csv", size="md", elem_classes=["download-btn"])
            btn_sql = gr.DownloadButton(label="Download sql", size="md", elem_classes=["download-btn"])
            btn_json = gr.DownloadButton(label="Download json", size="md", elem_classes=["download-btn"])

        chain = btn.click(
            fn=generate_mockup_data,
            inputs=[category, num_data_rows, columns, a_instruction],
            outputs=out,
            queue=True
        )

        chain = chain.then(
            fn=show_hidden_row,
            inputs=None,
            outputs=buttons_row,
        )

        btn_csv.click(
            lambda category, data: make_file("csv", category, data),
            inputs=[category, out],
            outputs=btn_csv
        )

        btn_sql.click(
            lambda category, data: make_file("sql", category, data),
            inputs=[category, out],
            outputs=btn_sql
        )

        btn_json.click(
            lambda category, data: make_file("json", category, data),
            inputs=[category, out],
            outputs=btn_json
        )

        ### Pre-warming the model right upon the page load
        ### in order to save the model load time when user submitting the form.
        demo.load(lambda: enable_model(), queue=False)

    return demo


if __name__ == "__main__":
    app = render_interface()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="0.0.0.0", server_port=7860)