import torch
import pickle

class PedalKeeperNet(torch.nn.Module):
    def __init__(self) -> None:
        super(PedalKeeperNet, self).__init__()
        embs = 128
        
        # L1
        # input : 1 channel, embs width, N batch
        # after conv : 64 channel, embs width, N batch
        # after pool : 64 channel, embs/2 width, N batch
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2)
        )

        # L2
        # input : 64 channel, embs/2 width, N batch
        # after conv : 128 channel, embs/4 width, N batch
        # after pool : 128 channel, embs/4 width, N batch
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2)
        )

        # FC
        # input : 128 channel * embs/4 height
        # output : 1 acc + 1 brk
        self.fc = torch.nn.Linear(128 * (embs / 4), 2)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print("현재 device : {}".format(device))

learning_rate = 0.001
training_epochs = 30

model = PedalKeeperNet().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

f = open('result.pckl', 'rb')
result = pickle.load(f)
batch = len(result['embs'])

print("총 학습 데이터 {0}개".format(batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for current_set in range(result['embs']):
        x = current_set
        x = x.unsqueeze(1)
        y = 0
        _, y = y.max(dim=1)
        
        optimizer.zero_grad()
        hypothesis = model(x)
        cost = criterion(hypothesis, y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / batch
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))