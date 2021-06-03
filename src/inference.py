import torch
from utils import readfile as rf
import numpy as np
from tqdm import tqdm
import os
from utils import init_model
from pyserini.search import SimpleSearcher


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
            trec_lines.append(
                str(qid) + "\t" + "Q0" + "\t" + str(docid) + "\t" + str(k + 1) + "\t" + str(score) + "\t" + "BERT" + "\n")

        with open(output_path, "a+") as f:
            f.writelines(trec_lines)


def inference_search(model, tokenizer, path_to_index, dev_query_path, res_path, qrel_path, output_path, isDataParallel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    rerank_cut = 1500

    model.eval().to(device)
    qrel = rf.read_qrel(qrel_path)
    run = rf.read_result_topK(res_path)
    queries = rf.read_queries_list(dev_query_path)
    # collection = rf.read_collection(collection_path)
    qid2indexes = {str(i): (0 if i < 80 else 1) for i in range(1, 118)}
    searcher = [SimpleSearcher(path_to_index[0]), SimpleSearcher(path_to_index[1])]

    def get_text(qid, docid):
        return searcher[qid2indexes[qid]].doc(docid).raw()

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
                batch_passages.append(get_text(qid, docid))

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
        scores = [i for i in range(len(docids), 0, -1)]
        # write run file
        write_run_file([qid], [scores], [docids], output_path)


def inference(model, tokenizer, collection_path, dev_query_path, res_path, qrel_path, output_path, isDataParallel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    rerank_cut = 1500

    model.eval().to(device)
    qrel = rf.read_qrel(qrel_path)
    run = rf.read_result_topK(res_path)
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
        scores = [i for i in range(len(docids), 0, -1)]
        # write run file
        write_run_file([qid], [scores], [docids], output_path)


def run_inference(pretrained, phase, eval_year, ir_method, path_to_trained_model, output):
    isFinetune = 1
    device = None

    # dev path
    dev_collection_path = "data/year/collection/{}_{}_{}.txt".format(eval_year, phase, ir_method)
    dev_dataset_path = "runs/{}_{}_{}/pyserini_dev_demofilter_bm25_1500_{}.res".format(eval_year, ir_method, phase,
                                                                                       phase)
    dev_query_path = "data/year/queries_{}_{}.txt".format(eval_year, phase)
    dev_qrel_path = "data/year/qrels_{}_{}.txt".format(eval_year, phase)

    # Initialization model
    device, tokenizer, model = init_model.model_init(device, pretrained, isFinetune, output, path_to_trained_model)

    bert_out_path = os.path.join(output, f'{pretrained}_{phase}_1500.res')
    # print(bert_out_path)
    inference(model, tokenizer, dev_collection_path, dev_query_path, dev_dataset_path, dev_qrel_path, bert_out_path, 1)


def run_neg_inference(pretrained, phase, eval_year, ir_method, path_to_trained_model, output):
    isFinetune = 1
    device = None

    # dev path
    # dev_collection_path = "data/year/collection/{}_{}_{}.txt".format(eval_year, phase, ir_method)
    dev_collection_path = ['../pyserini/indexes/TRECPM2017_txt_v2', '../pyserini/indexes/TRECPM2019_txt_v2']
    dev_dataset_path = "runs_bm25_neg/{}_{}_{}/pyserini_dev_bm25_10000_{}.res".format(eval_year, ir_method,
                                                                                      phase, phase)
    dev_query_path = "data/year/queries_{}_{}.txt".format(eval_year, phase)
    dev_qrel_path = "data/year/qrels_{}_{}.txt".format(eval_year, phase)

    # Initialization model
    device, tokenizer, model = init_model.model_init(device, pretrained, isFinetune, output, path_to_trained_model)

    bert_out_path = os.path.join(output, f'{pretrained}_{phase}_neg.res')
    # print(bert_out_path)
    inference_search(model, tokenizer, dev_collection_path, dev_query_path, dev_dataset_path, dev_qrel_path, bert_out_path, 1)

def run_inference_iteration(model, tokenizer, eval_year, ir_method, output, epoch):
    isFinetune = 1
    device = None
    phase = 'train'
    pretrained = 'BioBERT'

    # dev path
    dev_collection_path = "data/year/collection/{}_{}_{}.txt".format(eval_year, phase, ir_method)
    dev_dataset_path = "runs/{}_{}_{}/pyserini_dev_demofilter_bm25_1500_{}.res".format(eval_year, ir_method, phase,
                                                                                       phase)
    dev_query_path = "data/year/queries_{}_train.txt".format(eval_year, phase)
    dev_qrel_path = "data/year/qrels_{}_train.txt".format(eval_year, phase)

    bert_out_path = os.path.join(output, f'runs/{eval_year}_{epoch}_{pretrained}_train_1500.res')
    # print(bert_out_path)
    inference(model, tokenizer, dev_collection_path, dev_query_path, dev_dataset_path, dev_qrel_path, bert_out_path, 1)

def run_neg_inference(model, tokenizer, eval_year, ir_method, output, epoch):
    isFinetune = 1
    device = None
    phase = 'train'
    pretrained = 'BioBERT'

    # dev path
    dev_collection_path = ['../pyserini/indexes/TRECPM2017_txt_v2', '../pyserini/indexes/TRECPM2019_txt_v2']
    dev_dataset_path = "runs_bm25_neg/{}_{}_{}/pyserini_dev_bm25_10000_{}.res".format(eval_year, ir_method,
                                                                                      phase, phase)
    dev_query_path = "data/year/queries_{}_{}.txt".format(eval_year, phase)
    dev_qrel_path = "data/year/qrels_{}_{}.txt".format(eval_year, phase)

    bert_out_path = os.path.join(output, f'runs/{eval_year}_{epoch}_{pretrained}_train_neg.res')
    # print(bert_out_path)
    inference_search(model, tokenizer, dev_collection_path, dev_query_path, dev_dataset_path, dev_qrel_path, bert_out_path, 1)
