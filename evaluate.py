import fnmatch
import argparse
import pandas

from tqdm import tqdm
import numpy as np
import transformers


def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Genarate',
                    description='This sript generates answers for questions from csv file')

    parser.add_argument(
        "--metric",
        default="similarity",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )   
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
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


def evaluate_similarity(model, csv_gold, csv_prediction, csv_res_path):
    from sentence_transformers import util

    data_gold = pandas.read_csv(csv_gold)
    data_prediction = pandas.read_csv(csv_prediction)
    
    res = []
    answers_gold = data_gold['answers'].values
    answers_prediction = data_prediction['answers'].values


    for (gold, prediction) in tqdm(zip(answers_gold, answers_prediction)):
        embeddings = model.encode([gold, prediction])
        cos_sim = util.cos_sim(embeddings, embeddings)
        res.append(cos_sim[0, 1])

    print("Similarity: ", np.mean(res))

    if csv_res_path is not None:
        res_data = {'questions': list(data_gold['questions'].values), 'similarity': res}
        df = pandas.DataFrame(res_data)
        df.to_csv(csv_res_path)


def main():
    args = parse_args()

    if args.metric == 'similarity':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)

        evaluate_similarity(model, args.gold, args.prediction, args.save_evaluation_path)


if __name__ == "__main__":
    main()


# try sentence-transformers/all-mpnet-base-v2
