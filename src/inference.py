import torch
from utils import readfile as rf
import numpy as np
from tqdm import tqdm
import os


def write_run_file(qids, scores, docids, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    num_quries = len(qids)
    for j in range(num_quries):
        qid = qids[j]
        num_docs = len(docids[j])
        trec_lines = []
        for k in range(num_docs):
            score = scores[j][k]
            docid = docids[j][k]
            trec_lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(k + 1) + " " + str(score) + " " + "BERT" + "\n")

        with open(output_path, "a+") as f:
            f.writelines(trec_lines)


def inference(model, tokenizer, collection_path, dev_query_path, res_path, qrel_path, output_path, isDataParallel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    rerank_cut = 50

    model.eval().to(device)
    qrel = rf.read_qrel(qrel_path)
    run = rf.read_result_topK(res_path, qrel.keys())
    queries = rf.read_queries_list(dev_query_path)
    collection = rf.read_collection(collection_path)

    # clear out file
    with open(output_path, "w") as f:
        f.writelines('')

    for qid in tqdm(qrel.keys(), desc="Ranking queries...."):
        query = queries[qid]
        # split batch of documents in top 1000
        docids = run[qid]
        num_docs = min(rerank_cut, len(docids))  # rerank top k
        numIter = num_docs // batch_size + 1

        total_scores = []
        for i in range(numIter):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > num_docs:
                end = num_docs
                if start == end:
                    continue

            batch_passages = []
            for docid in docids[start:end]:
                batch_passages.append(collection[docid])

            inputs = tokenizer([query] * len(batch_passages), batch_passages,
                               return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(device)
            with torch.no_grad():
                if isDataParallel:
                    scores = model.module.get_scores(inputs)
                else:
                    scores = model.get_scores(inputs)
                total_scores.append(scores)

        total_scores = torch.cat(total_scores).cpu().numpy()
        # rerank documents
        zipped_lists = zip(total_scores, docids)
        sorted_pairs = np.array(sorted(zipped_lists, reverse=True))
        # scores = sorted_pairs[:, 0]
        docids[:num_docs] = sorted_pairs[:, 1]
        scores = [i for i in range(len(docids),0,-1)]
        # write run file
        write_run_file([qid], [scores], [docids], output_path)
