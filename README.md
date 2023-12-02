# Simple benchmark for evaluating quality of compressed LLMs.

Simple and short test for evaluating quality of compressed, quantized, pruned LLMs.
The best configuration is not guaranteed.

## Description

Implementation of short tests for understanding similarity of text generation between original and comperssed LLMs.

## Getting Started

### Dependencies

* See requirements.txt

### Installing

* git clone https://github.com/andreyanufr/who_what_benchmark.git
* python -m venv eval_env
* source eval_env/bin/activate
* pip install -r requirements.txt

### Executing program

* How to run the program
* Step-by-step bullets
```
# run text generation for original model
python3 generate.py --modeltype causal --model meta-llama/Llama-2-7b-chat-hf --save_generations_path gold_llama-2-7b-chat-hf.csv --csv simple.csv --trust_remote_code

# convert and compress llama with optimum-intel and nncf and save it to some folder
...

#run text generation for compressed models
python3 generate.py --modeltype ov_causal --model /home/user/models/meta-llama/Llama-2-7b-chat-hf-int8 --save_generations_path predict_llama-2-7b-chat-hf_int8.csv --csv simple.csv --trust_remote_code

python3 generate.py --modeltype ov_causal --model /home/user/models/meta-llama/Llama-2-7b-chat-hf-int4_sym --save_generations_path predict_llama-2-7b-chat-hf_int4_sym.csv --csv simple.csv --trust_remote_code

python3 generate.py --modeltype ov_causal --model /home/user/models/meta-llama/Llama-2-7b-chat-hf-int4_asym --save_generations_path predict_llama-2-7b-chat-hf_int4_asym.csv --csv simple.csv --trust_remote_code


for file in predict_llama-2-7b*; do
python3 evaluate.py --gold gold_llama-2-7b-chat-hf.csv --prediction $file --save_evaluation_path eval_$file 2>&1 | tee -a eval.log
done
```

### Notes

* In the file save_evaluation_path you can see per sample similarity metrics.
* Input CSV file for generation must contain column with name `questions`. For example see simple.csv
* You can see example of generation in file generations.csv
* evaluate.py uses for similarity measurement [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) but you can use other similar network.
