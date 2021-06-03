import torch
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW
import matplotlib.pyplot as plt

def test_warmup():
    train_len = 100
    num_epochs =2
    lr=2e-5
    num_warmup_steps = int(train_len * 0.1)
    num_training_steps = int(train_len) * 0.9 + 1 + train_len * (num_epochs - 1)
    model = torch.nn.Linear(100,1)
    optim = AdamW(model.parameters(), lr=lr, correct_bias=False)  # 2e-5
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_training_steps)
    res = []
    for _ in range(1):
        for i in range(train_len):
            scheduler.step()
            res.append(optim.param_groups[0]['lr'])

    print(optim.state_dict())
    print(scheduler.state_dict())
    for _ in range(1):
        for i in range(train_len):
            scheduler.step()
            res.append(optim.param_groups[0]['lr'])

    plt.plot([i for i in range(train_len*num_epochs)], res)
    plt.savefig('test/lr_test.png')
