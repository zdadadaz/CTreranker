from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold


def split(iscv, isFinetune):
    if iscv == 'cv':
        return split_cv(5)
    else:
        return split_year(iscv, isFinetune)


def split_cv(numfold):
    topics = np.array([i for i in range(1, 118)])
    kf = KFold(n_splits=numfold, shuffle=True)
    res = []
    for train_index, test_index in kf.split(topics):
        dataidx = {'train': topics[train_index],'val':[],'test':topics[test_index]}
        res.append(dataidx)
    return res


def split_year(eval_year, isFinetune):
    # train 17, 18, test 19
     # qbyyear = [[1, 29], [30, 79], [80, 117]]
    val_idx = []
    if eval_year == '2019':
        train_idx = [str(i) for i in range(1,80)]
        test_idx = [str(i) for i in range(80,118)]
    elif eval_year == '2018':
        train_idx = [str(i) for i in range(1, 30) ] + [str(i) for i in range(80, 118)] # 81
        test_idx = [str(i) for i in range(30, 80)]  # 121
    else: # 2017
        train_idx = [str(i) for i in range(30, 80)] + [str(i) for i in range(80, 118)]  # 81
        test_idx = [str(i) for i in range(1, 30)]  # 121

    if not isFinetune:
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=1)

    dataidx = {'train': train_idx,'val':val_idx,'test':test_idx}
    return dataidx

def split_ts(qids):
    val_idx = []
    train_idx = qids
    train_idx, test_idx = train_test_split(train_idx, test_size=0.2, random_state=1)
    dataidx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    return dataidx

def split_ts_cv(qids, numfold):
    topics = np.array(qids)
    kf = KFold(n_splits=numfold, shuffle=True, random_state=0)
    res = []
    for train_index, test_index in kf.split(topics):
        dataidx = {'train': topics[train_index],'val':[],'test':topics[test_index]}
        res.append(dataidx)
    return res