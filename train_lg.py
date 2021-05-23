from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from src.pl_callbacks import CheckpointEveryEpoch
import torch
from src.model import BertReranker
from src.datagen import TrecPMTrainset
import os
from utils.arg import argretrieve
from src.inference import inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = None

def collate_fn(batch):
    queries = []
    passages = []
    labels = []
    global tokenizer
    for query, passage, label in batch:
        queries.append(query)
        passages.append(passage)
        labels.append(label)
    inputs = tokenizer(queries, passages, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    return inputs, torch.stack(labels)


def main():
    seed_everything(313)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = argretrieve()
    model_type = "bert-base-uncased"
    cache_dir = "../../cache"
    year = '2019'
    irmethod = 'bm25'
    # train path
    collection_path = "data/year/collection/{}_train_{}.txt".format(year, irmethod)
    dataset_path = "runs/{}_bm25_train/pyserini_dev_demofilter_bm25_1000_train.res".format(year)
    path_to_pos_qrel = "data/year/pos_qrels_{}_train.txt".format(year)
    path_to_query = "data/year/queries_{}_train.txt".format(year)
    # dev path
    dev_collection_path = "data/year/collection/{}_test_{}.txt".format(year, irmethod)
    dev_dataset_path = "runs/{}_bm25_test/pyserini_dev_demofilter_bm25_1000_test.res".format(year)
    dev_query_path = "data/year/queries_{}_test.txt".format(year)
    dev_qrel_path = "data/year/qrels_{}_test.txt".format(year)
    dev_res_path = "output_lg/{}/runs/run.trec{}-bm25".format(model_type, year)

    # parameters
    batch_size = 32
    lr = 2e-5
    optimizer = 'adam'
    warm_up_steps_percent = 10
    gpus_per_node = 1
    num_nodes = 1
    num_epochs = 1
    num_neg_per_pos = 4
    num_gpus = gpus_per_node * num_nodes
    log_name = "{}_{}_bs{}_gpu{}_bm25_top1000_{}_lr{}_warm{}".format(year, model_type, batch_size, gpus_per_node,
                                                                     optimizer, lr, warm_up_steps_percent)
    save_path = "output_lg/{}/ckpts/{}".format(model_type, log_name)

    tb_logger = pl_loggers.TensorBoardLogger('output_lg/{}/logs/'.format(model_type), name=log_name)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(model_type, cache_dir=cache_dir)

    gradient_checkpointing = True  # trade-off training speed for batch_size per GPU
    config.gradient_checkpointing = gradient_checkpointing

    train_set = TrecPMTrainset(dataset_path,
                               path_to_pos_qrel,
                               path_to_query,
                               collection_path,
                               tokenizer,
                               num_neg_per_pos=num_neg_per_pos,
                               num_epochs=num_epochs)

    print("Training set size:", len(train_set))
    print("save_path", save_path)
    callbacks = [CheckpointEveryEpoch(1, 1, 1, save_path),
                 LearningRateMonitor(logging_interval='step')]

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  shuffle=True,
                                  num_workers=10,
                                  collate_fn=collate_fn
                                  )

    model = BertReranker(encoder_name_or_dir=model_type,
                         encoder_config=config,
                         cache_dir=cache_dir,
                         lr=lr,
                         warm_up_steps_percent=warm_up_steps_percent,
                         num_gpus=num_gpus,
                         batch_size=batch_size,
                         train_set_size=len(train_set),
                         num_epochs=1,  # this is fake num_epoch.
                         num_neg_per_pos=num_neg_per_pos,
                         )

    trainer = Trainer(max_epochs=1,
                      gpus=gpus_per_node,
                      num_nodes=num_nodes,
                      checkpoint_callback=False,
                      logger=tb_logger,
                      # amp_backend='apex',
                      # amp_level='O1',
                      # accelerator="ddp",
                      # plugins='ddp_sharded',
                      log_every_n_steps=10,
                      callbacks=callbacks,
                      )

    trainer.fit(model, train_dataloader)

    # print("Loading model...")
    # model = BertReranker.load_from_checkpoint(checkpoint_path=save_path+'_epoch1.ckpt')
    # inference(model, tokenizer, dev_collection_path, dev_query_path, dev_dataset_path, dev_qrel_path, dev_res_path)

if __name__ == '__main__':
    main()
