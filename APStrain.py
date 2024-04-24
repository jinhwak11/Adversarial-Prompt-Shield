from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from utils.model import DistilBERTClassifier
from utils.dataloader import DataPreprocessing
from torch import cuda
import argparse
import os
from tqdm import tqdm

device = 'cuda' if cuda.is_available() else 'cpu'
print("DEVICE", device)

### TRAINING HYPERPARAMETER SETUP ###
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-05
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

def parser_args():
    '''
    : Arguments
            training_dt  (str, sep=','): dataname to train the model (e.g. wtc,bbf_s,bbf_a,bbf_multi,bad)
            datapath (str): path where data is stored in
            multiturn (int) : number of turns to be used for preprocessing the multi-turn dialogue data
    '''
    parser = argparse.ArgumentParser(description="dataname, model_path")
    parser.add_argument("--training_dt", type=str)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--model_savepath", type=str)
    parser.add_argument('--multiturn',type=int,default=8)
    args = parser.parse_args()
    return args

class PrepData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,multi_turn):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multi_turn = multi_turn

    def __getitem__(self, index):
        dialogue = str(self.data.transcript[index])
        dialogue = " ".join(dialogue.split("\n\n")[1:][-self.multi_turn:]) # use only last n-turns of dialogues.
        inputs = self.tokenizer.encode_plus(
            dialogue,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.int),
            'mask': torch.tensor(mask, dtype=torch.int),
            'targets': torch.tensor(self.data.rating[index], dtype=torch.float)
        }

    def __len__(self):
        return self.len


def calculate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def valid(model, testing_loader):
    sigmoid = torch.nn.Sigmoid()
    model.eval()
    te_loss = 0
    all_predictions = []
    all_targets = []
    n_correct = 0;
    n_wrong = 0;
    total = 0;
    nb_tr_steps = 0;
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float).reshape(-1, 1)
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            te_loss += loss.item()
            outputs = sigmoid(outputs)
            big_idx = torch.round(outputs)
            all_predictions.extend(list(big_idx))
            all_targets.extend(list(targets))
            n_correct += calculate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
    epoch_loss = te_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    precision = sum([p == l for p, l in zip(all_predictions, all_targets) if l > 0]) / sum(all_predictions)
    recall = sum([p == l for p, l in zip(all_predictions, all_targets) if l > 0]) / sum(all_targets)
    print(f"Precision {precision} , Recall {recall}")
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu, all_predictions, all_targets

import copy
class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.max_validation_f1 = 0
        self.model = None

    def early_stop(self, model, validation_f1):
        if validation_f1 >= self.max_validation_f1:
            self.max_validation_f1 = validation_f1
            self.counter = 0
            self.model = copy.deepcopy(model)
        elif validation_f1 < self.max_validation_f1:
            self.counter += 1

def train(epoch):
    sigmoid = torch.nn.Sigmoid()
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float).reshape(-1, 1)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        outputs = sigmoid(outputs)
        big_idx = torch.round(outputs)
        # big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 2000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 2000 steps: {loss_step}")
            print(f"Training Accuracy per 2000 steps: {accu_step}")
            writer.add_scalars('Training Loss and Acc step ',
                               {'Training loss': loss_step, 'Training Accuracy': accu_step},
                               epoch * len(training_loader) + _)
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    valid_acc, valid_predictions, valid_targets = valid(model, valid_loader)
    valid_precision = sum([p == l for p, l in zip(valid_predictions, valid_targets) if l > 0]) / sum(valid_predictions)
    valid_recall = sum([p == l for p, l in zip(valid_predictions, valid_targets) if l > 0]) / sum(valid_targets)
    valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall)
    writer.add_scalars('Valid Acc, Unsafe Precision, Recall  ',
                       {'Valid Acc': valid_acc, 'Valid Precision': valid_precision, 'Valid Recall': valid_recall,
                        'Valid f1': valid_f1},
                       epoch)
    early_stopper.early_stop(model, valid_f1)
    writer.flush()
    return

if __name__ == "__main__":
    # terminal command example
    # python APStrain.py --training_dt wtc,bbf_s,bbf_a,bbf_multi,bad --datapath /mnt/llm/data/ --model_savepath /mnt/llm/models/

    args = parser_args()
    import datetime

    dayinfo = datetime.datetime.now()
    day_path = dayinfo.strftime("%m%d")
    MULTI_TURN = args.multiturn
    dataprep = DataPreprocessing(dataname=args.training_dt,data_folder_path=args.datapath)
    total_train_data = dataprep.read_and_processing('train')
    total_valid_data = dataprep.read_and_processing('valid')
    print("TOTAL TRAIN DATA", total_train_data.shape, "TOTAL VALID DATA", total_valid_data.shape)
    from collections import Counter
    print("TOTAL TRAIN case ratio:",Counter(total_train_data['rating']))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    writer = SummaryWriter(f'./logs/{day_path}distill_training_{args.training_dt}_epoch{EPOCHS}')
    training_set = PrepData(total_train_data, tokenizer, MAX_LEN,multi_turn=MULTI_TURN)
    valid_set = PrepData(total_valid_data, tokenizer, MAX_LEN,multi_turn=MULTI_TURN)
    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **train_params)
    model = DistilBERTClassifier()
    model = torch.nn.DataParallel(model, device_ids=[0,1]) # if you want to run training with multi-gpus, utilize this command.
    model.to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]).to(device))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopper(patience=3)

    ## Saving the files for re-use
    if not os.path.exists(f'{args.model_savepath}/{day_path}distil_{args.training_dt}_epoch{EPOCHS}_multiturn{MULTI_TURN}'):
        os.mkdir(f'{args.model_savepath}/{day_path}distil_{args.training_dt}_epoch{EPOCHS}_multiturn{MULTI_TURN}')
    output_model_file = f'{args.model_savepath}/{day_path}distil_{args.training_dt}_epoch{EPOCHS}_multiturn{MULTI_TURN}/pytorch_distilbert_cls.bin'
    output_vocab_file = f'{args.model_savepath}/{day_path}distil_{args.training_dt}_epoch{EPOCHS}_multiturn{MULTI_TURN}/'

    for epoch in tqdm(range(EPOCHS)):
        train(epoch)
        torch.save(model.state_dict(), output_model_file)
        tokenizer.save_pretrained(output_vocab_file)
        if early_stopper.patience <= early_stopper.counter:
            break
