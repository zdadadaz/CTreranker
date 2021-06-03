from src.inference import run_inference, run_neg_inference
import pathlib
import os

def run_inference_test():
    pretrained = 'BioBERT'
    phase = 'train'

    # checkpoint_path = 'output_exclude_positive_include_all_topk/2019_bm25_BERT_length256_neg10_v2_kw_pretrained_BioBERT/best_ft.pt'
    ir_method = 'bm25'
    for eval_year in [2017, 2018, 2019]:
        checkpoint_path = f'output/{eval_year}_bm25_BERT_length256_neg10_baseline_pretrained_BioBERT/best_ft.pt'
        output = os.path.join("runs", "{}_{}_{}".format(eval_year, pretrained, phase))
        pathlib.Path(output).mkdir(parents=True, exist_ok=True)
        run_inference(pretrained, phase, eval_year, ir_method, checkpoint_path, output)

def run_inference_neg_test():
    pretrained = 'BioBERT'
    phase = 'train'
    # checkpoint_path = 'output_exclude_positive_include_all_topk/2019_bm25_BERT_length256_neg10_v2_kw_pretrained_BioBERT/best_ft.pt'
    ir_method = 'bm25'
    for eval_year in [2017, 2018, 2019]:
        checkpoint_path = f'output/{eval_year}_bm25_BERT_length256_neg10_baseline_pretrained_BioBERT/best_ft.pt'
        output = os.path.join("runs_bert_neg", "{}_{}_{}".format(eval_year, pretrained, phase))
        pathlib.Path(output).mkdir(parents=True, exist_ok=True)
        run_neg_inference(pretrained, phase, eval_year, ir_method, checkpoint_path, output)
