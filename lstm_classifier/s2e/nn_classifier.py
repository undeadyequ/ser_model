import torch
import sys
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import load_data, evaluate, plot_confusion_matrix

from config import dnn_model_config
from config import revdnn_model_config


class DNNClassifier(nn.Module):
    def __init__(self, idim, edim, units=128, layers=2):

        super(DNNClassifier, self).__init__()
        self.ser = torch.nn.ModuleList()
        for layer in range(layers):
            ichans = idim if layer == 0 else units
            ochans = edim if layer == layers - 1 else units
            if layer != layers - 1:
                ser = torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                )
            else:
                ser = torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.Softmax(dim=1)
                )
            self.ser += [ser]

    def forward(self, emo_feats):

        for i in range(len(self.ser)):
            emo_feats = self.ser[i](emo_feats)

        return emo_feats

    def inference(self, emo_feats: torch.Tensor):
        emo_feats = emo_feats.unsqueeze(0)
        return self.forward(emo_feats)


def train(model, config):

    device = 'cuda:{}'.format(config['gpu']) if \
             torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_batches = load_data()
    test_pairs = load_data(test=True)

    best_acc = 0
    for epoch in range(config['n_epochs']):
        losses = []
        for batch in train_batches:
            inputs = batch[0]  # frame in format as expected by model
            targets = batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)
            predictions = predictions.to(device)

            loss = criterion(predictions, targets)


            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # evaluate
        with torch.no_grad():
            inputs = test_pairs[0]
            targets = test_pairs[1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = torch.argmax(model(inputs), dim=1)  # take argmax to get class id
            predictions = predictions.to(device)

            # evaluate on cpu
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())

            # Get results
            # plot_confusion_matrix(targets, predictions,
            #                       classes=emotion_dict.keys())
            performance = evaluate(targets, predictions)
            if performance['acc'] > best_acc:
                best_acc = performance['acc']
                print(performance)
                # save model and results
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, 'runs/{}-best_model.pth'.format(config['model_code']))

                with open('results/{}-best_performance.pkl'.format(config['model_code']), 'wb') as f:
                    pickle.dump(performance, f)

if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}
    #model = DNNClassifier(idim=8, edim=6)

    model = DNNClassifier(idim=8, edim=6)

    train(model, revdnn_model_config)
