import fnmatch
import json
import argparse
import pandas

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

try:
    from optimum.intel import OVModelForCausalLM
except ImportError:
    print("Not import optimum.intel")


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Genarate',
                    description='This sript generates answers for questions from csv file')

    parser.add_argument(
        "--model",
        default="stabilityai/stablelm-3b-4e1t",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--csv",
        default='simple.csv',
        help="CSV file with questions. Must have column with name questions."
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.csv",
        help="Path for saving the code generations",
    )
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def generate(model, tokenizer, csv_name, out_name):
    data = pandas.read_csv(csv_name)
    
    res = []
    questions = data['questions']
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=640)
    
    answers = []
    
    for q in questions.values:
        out = pipe(q)
        out = out[0]['generated_text']
        answers.append(out[len(q):])

    
    dict = {'questions': list(questions.values), 'answers': answers}
    df = pandas.DataFrame(dict)
    df.to_csv(out_name)


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    accelerator = Accelerator()


    # here we generate code and save it (evaluation is optional but True by default)
    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.precision not in dict_precisions:
        raise ValueError(
            f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
        )

    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "use_auth_token": args.use_auth_token,
    }
    if args.load_in_8bit:
        print("Loading model in 8bit")
        model_kwargs["load_in_8bit"] = args.load_in_8bit
        model_kwargs["device_map"] = {"": accelerator.process_index}
    elif args.load_in_4bit:
        print("Loading model in 4bit")
        model_kwargs["load_in_4bit"] = args.load_in_4bit
        model_kwargs["device_map"] = {"": accelerator.process_index}
    elif args.modeltype != 'ov_causal':
        print(f"Loading model in {args.precision}")
        model_kwargs["torch_dtype"] = dict_precisions[args.precision]


    if args.modeltype == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )
    elif args.modeltype == 'ov_causal':
        model = OVModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )
    else:
        raise ValueError(
            f"Non valid modeltype {args.modeltype}, choose from: causal, ov_causal"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
        truncation_side="left",
        padding_side="right",  # padding on the right is needed to cut off padding in `complete_code`
    )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    try:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Some models like CodeGeeX2 have pad_token as a read-only property
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass
    WIZARD_LLAMA_MODELS = [
        "WizardLM/WizardCoder-Python-34B-V1.0",
        "WizardLM/WizardCoder-34B-V1.0",
        "WizardLM/WizardCoder-Python-13B-V1.0"
    ]
    if args.model in WIZARD_LLAMA_MODELS:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = 1
        print("Changing bos_token to <s>")

    generate(model, tokenizer, args.csv, args.save_generations_path)


if __name__ == "__main__":
    main()


# try sentence-transformers/all-mpnet-base-v2
