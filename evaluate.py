import argparse
import pandas as pd

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from difflib import SequenceMatcher

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Generate',
                    description='This sript generates answers for questions from csv file')

    parser.add_argument(
        "--metrics",
        default="[similarity, divergency]",
        nargs='+',
        choices=["similarity", "divergency"],
        help="""Metric for evaluation between ground truth output and predicted one:\n
            \"similarity\" refers to the cosine similarity between the given outputs, encoded by some neural network\n
            \"divergency\" refers to the divergent token metrics (https://arxiv.org/abs/2311.01544), where\n
            FDT - Average position of the first divergent token. The worst is 0.\n
            FDT norm - Average share of matched tokens until first divergent one. The best is 1.\n
            SDT - Average number of divergent tokens in the evaluated outputs. The best is 0.\n
            SDT norm - Average share of divergent tokens in the evaluated outputs. The best is 0, the maximum is 1.\n
        """
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="meta-llama/Llama-2-7b",
        help="Tokenizer for divergency metric. Provide a repo name in Hugging Face hub or a local path",
    )

    parser.add_argument(
        "--gold",
        default=None,
        help="CSV file with ground truth."
    )
    parser.add_argument(
        "--prediction",
        default=None,
        help="CSV file with predictions."
    )
    parser.add_argument(
        "--save_evaluation_path",
        type=str,
        default="eval.csv",
        help="Path for saving per sample metrics in CSV format",
    )

    return parser.parse_args()


def evaluate_similarity(model, data_gold, data_prediction ):
    from sentence_transformers import util

    answers_gold = data_gold['answers'].values
    answers_prediction = data_prediction['answers'].values

    metric_per_question = []
    for (gold, prediction) in tqdm(zip(answers_gold, answers_prediction)):
        embeddings = model.encode([gold, prediction])
        cos_sim = util.cos_sim(embeddings, embeddings)
        metric_per_question.append(cos_sim[0, 1].item())

    metric_dict = {'similarity': np.mean(metric_per_question)}
    return metric_dict, {'similarity': metric_per_question}

def evaluate_divergency(tokenizer, data_gold, data_prediction ):
    answers_gold = data_gold['answers'].values
    answers_prediction = data_prediction['answers'].values

    DEBUG = False
    # NOTE: a - reference answers, b - answers to evaluate
    fdt_list, sdt_list, sdtn_list, fdt_max = [], [], [], []

    for a_answer, b_answer in zip(answers_gold, answers_prediction):
        a_indexes = tokenizer.encode(a_answer, return_tensors="pt").squeeze().tolist()
        b_indexes = tokenizer.encode(b_answer, return_tensors="pt").squeeze().tolist()
        fdt_max.append(len(a_indexes))

        matcher = SequenceMatcher(None, a_indexes, b_indexes)
        blocks = matcher.get_matching_blocks()
        a, b, size = blocks[0]
        fdt = 0
        if a == 0 and b == 0:
            fdt = blocks[0].size
        fdt_list.append(fdt)

        num_matched = sum(block.size for block in blocks)
        sdt = len(b_indexes) - num_matched  # how many tokens to correct in the prediction
        sdt_list.append(sdt)
        sdt_norm = sdt / len(b_indexes)  # share of tokens to correct in the prediction
        sdtn_list.append(sdt_norm)

        if DEBUG:
            print(blocks)
            for block in blocks:
                a, b, size = block
                matched = a_indexes[a : a + size + 1]
                print(matched)
                print(tokenizer.decode(matched))
                matched = b_indexes[b : b + size + 1]
                print(matched)
                print(tokenizer.decode(matched))

    fdt_max = np.average(fdt_max)
    metric_per_question = {
        'FDT': fdt_list,
        'SDT': sdt_list,
        'FDT norm': np.array(fdt_list) / fdt_max,
        'SDT norm': sdtn_list
    }

    fdt_avg = np.average(fdt_list)
    metric_dict = {
        'FDT': fdt_avg,
        'SDT': np.average(sdt_list),
        'FDT norm': fdt_avg / fdt_max,
        'SDT norm': np.average(sdtn_list),
    }

    return metric_dict, metric_per_question



def main():
    args = parse_args()
    metrics = args.metrics

    data_gold = pd.read_csv(args.gold)
    data_prediction = pd.read_csv(args.prediction)

    all_metrics_per_question = {}
    all_metrics = {}
    if 'similarity' in metrics:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)
        metric_dict, metric_per_question = evaluate_similarity(model, data_gold, data_prediction)
        all_metrics.update(metric_dict)
        all_metrics_per_question.update(metric_per_question)
    if 'divergency' in metrics:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            revision=None,
            trust_remote_code=True,
            use_auth_token=True,
            truncation_side="left",
            padding_side="right",  # padding on the right is needed to cut off padding in `complete_code`
        )
        metric_dict, metric_per_question = evaluate_divergency(tokenizer, data_gold, data_prediction)
        all_metrics.update(metric_dict)
        all_metrics_per_question.update(metric_per_question)

    print(pd.DataFrame([all_metrics]))

    if args.save_evaluation_path is not None:
        res_data = {'questions': list(data_gold['questions'].values), **all_metrics_per_question}
        df = pd.DataFrame(res_data)
        print(df)
        df.to_csv(args.save_evaluation_path)

if __name__ == "__main__":
    main()
