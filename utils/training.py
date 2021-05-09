import torch
import math
import os
import time
import sklearn
import sklearn.metrics
import tqdm
import numpy as np
from utils import readfile as rf
from utils import writefile as wf

class trainer():
    def __init__(self, model, output, batch_size, num_epochs, dataloaders, device, lr_step_period=None):
        self.model = model
        self.output = output
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataloaders = dataloaders
        self.device = device
        self.optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        if lr_step_period is None:
            lr_step_period = math.inf
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, lr_step_period)

    def _train_init_(self, f):
        epoch_resume = 0
        bestLoss = float("inf")
        # read previous trained model
        try:
            if self.device.type == 'cuda':
                checkpoint = torch.load(os.path.join(self.output, "checkpoint.pt"))
            else:
                checkpoint = torch.load(os.path.join(self.output, "checkpoint.pt"), map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['opt_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
            return epoch_resume, bestLoss
        except FileNotFoundError:
            f.write("Starting run from scratch\n")
            return epoch_resume, bestLoss

    def train(self):
        with open(os.path.join(self.output, "log.csv"), "a") as f:
            epoch_resume, bestLoss = self._train_init_(f)
            # Train one epoch
            for epoch in range(epoch_resume, self.num_epochs):
                print("Epoch #{}".format(epoch), flush=True)
                for phase in ['train', 'val']:
                    if phase not in self.dataloaders:
                        continue
                    start_time = time.time()
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_max_memory_allocated(i)
                        torch.cuda.reset_max_memory_cached(i)
                    loss, yhat, y, _ = self.run_epoch(self.model, self.dataloaders[phase], phase, self.optim, self.device)
                    f.write("{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                 phase,
                                                                 loss,
                                                                 sklearn.metrics.r2_score(yhat, y),
                                                                 time.time() - start_time,
                                                                 y.size,
                                                                 self.batch_size,
                                                                 self.optim.param_groups[0]['lr']
                                                                 ))
                    f.flush()

                self.scheduler.step()

                save = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': bestLoss,
                    'loss': loss,
                    'r2': sklearn.metrics.r2_score(yhat, y),
                    'opt_dict': self.optim.state_dict(),
                    'scheduler_dict': self.scheduler.state_dict(),
                }
                torch.save(save, os.path.join(self.output, "checkpoint.pt"))
                if loss < bestLoss:
                    torch.save(save, os.path.join(self.output, "best.pt"))
                    bestLoss = loss

            checkpoint = torch.load(os.path.join(self.output, "best.pt"))
            self.model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

    def run_epoch(self, model, dataloader, phase, optim, device):
        # criterion = torch.nn.MSELoss()  # Standard L2 loss
        criterion = torch.nn.BCEWithLogitsLoss()
        runningloss = 0.0
        if phase == 'train':
            self.model.train(phase == 'train')
        else:
            model.eval()
        counter = 0
        yhat = []
        y = []
        qdoc = []
        with torch.set_grad_enabled(phase == 'train'):
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for (i, item) in enumerate(dataloader):
                    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    # decoded = tokenizer.decode(item["input_ids"].numpy()[0])
                    # print(decoded)
                    # return
                    y.append(item['labels'].numpy())
                    qdoc.append(np.array(item['qdoc']))

                    X = item['input_ids'].to(device)
                    outcome = item['labels'].to(device)
                    attention_mask = item['attention_mask'].to(device)
                    outputs = model(X, attention_mask=attention_mask)
                    yhat.append(outputs.logits.to("cpu").detach().numpy())
                    loss = criterion(outputs.logits.float(), outcome.unsqueeze(1).float())
                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    runningloss += loss.item() * X.size(0)
                    counter += X.size(0)
                    epoch_loss = runningloss / counter

                    pbar.set_postfix_str("{:.2f} {:.2f}".format(epoch_loss, loss.item()))
                    pbar.update()

        y = np.concatenate(y)
        yhat = np.concatenate(yhat)
        qdoc = np.concatenate(qdoc)
        return epoch_loss, yhat, y, qdoc

    def test(self, qids, qrels, path_to_test_result, path_to_qrels, k, input_run_name, output_run_name):
        checkpoint = torch.load(os.path.join(self.output, "best.pt"))
        self.model.load_state_dict(checkpoint['state_dict'])
        with open(os.path.join(self.output, "log.csv"), "a") as f:
            for phase in ["val", "test"]:
                if phase in self.dataloaders:
                    loss, yhat, y, qdoc = self.run_epoch(self.model, self.dataloaders[phase], phase, None, self.device)
                    f.write("{} R2:   {:.3f} {:.3f} \n".format(phase, loss, sklearn.metrics.r2_score(yhat, y)))
                    f.write("{} MAE:  {:.2f} {:.2f} \n".format(phase, loss, sklearn.metrics.mean_absolute_error(yhat,y)))
                    f.flush()
        org_res = rf.read_result(path_to_test_result)
        rerank = self._gen_trec_ouput( yhat, qdoc, k)
        out_path = path_to_test_result.replace(input_run_name,output_run_name)[:-4]
        wf.write_rerank_res(org_res, rerank, k, output_run_name, out_path)
        wf.write_qrels(qids, qrels, path_to_qrels)

    def _gen_trec_ouput(self, yhat, qdoc, k):
        res = {}
        for i in range(0,len(yhat),k):
            qid = qdoc[i].split('NCT')[0]
            res[qid] = {'docid':[], 'rank':[], 'score':[]}
            idxlist = sorted(range(i,(i+k)), key=lambda a: -yhat[a])
            cnt = 1
            for j in idxlist:
                score = float(yhat[j])
                docid = 'NCT'+qdoc[j].split('NCT')[1]
                res[qid]['docid'].append(docid)
                res[qid]['rank'].append(str(cnt))
                res[qid]['score'].append(str(round(score,4)))
                cnt+=1
        return res