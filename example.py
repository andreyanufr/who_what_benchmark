from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    pipeline,
)

import whowhatbench

max_new_tokens = 128
model_small_id = "facebook/opt-125m"
model_small = AutoModelForCausalLM.from_pretrained(model_small_id)
tokenizer_small = AutoTokenizer.from_pretrained(model_small_id)

pipe_small = pipeline('text-generation', model=model_small, tokenizer=tokenizer_small, max_new_tokens=max_new_tokens)

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

evaluator = whowhatbench.Evaluator(tokenizer=tokenizer, text_gen_pipeline=pipe, test_data_path="/home/aanuf/proj/who_what_benchmark/simple.csv")

all_metrics_per_question, all_metrics = evaluator.score(pipe_small)

print(all_metrics_per_question)
print(all_metrics)
