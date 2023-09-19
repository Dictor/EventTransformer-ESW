import torch
import pickle5 as pickle
from torchsummary import summary
import math
from matplotlib import pyplot as plt

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
            torch.nn.AvgPool1d(2, 2)
        )

        # L2
        # input : 64 channel, embs/2 width, N batch
        # after conv : 128 channel, embs/4 width, N batch
        # after pool : 128 channel, embs/4 width, N batch
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2, 2)
        )

        # FC
        # input : 128 channel * embs/4 height
        # output : 1 acc + 1 brk
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * int(embs / 4),  1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2)
        )

        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        torch.nn.init.xavier_uniform_(self.fc[2].weight)

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
summary(model, ( 1, 128))
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

path = input("give pickle directory : ")
f = open(path + '/result.pckl', 'rb')
result = pickle.load(f)
f = open(path + '/result_pedal.pckl', 'rb')
result_pedal = pickle.load(f)

batch = len(result['embs'])
train_set_ratio = 0.75
train_set_len = int(math.floor(batch * train_set_ratio))
test_set_len = batch - train_set_len

print("총 학습 데이터 {0}개, train {1}개".format(batch, train_set_len))

evaluation_result = {"train": [], "test": []}
for epoch in range(training_epochs):
    avg_train_cost = 0
    avg_test_cost = 0
    
    for i in range(train_set_len):
        x = result['embs'][i]
        x = x.unsqueeze(1)
        y = torch.tensor([[result_pedal['accs'][i], result_pedal['brks'][i]]], dtype=torch.float32)
        
        optimizer.zero_grad()
        hypothesis = model(x)
        cost = criterion(hypothesis, y).float()
        cost.backward()
        optimizer.step()
        avg_train_cost += cost / batch
    
    with torch.no_grad():
        model.eval()
        test_loss = 0

        for i in range(test_set_len):
            x = result['embs'][train_set_len + i].to(device).float()
            x = x.unsqueeze(1)
            y = torch.tensor([[result_pedal['accs'][train_set_len + i], result_pedal['brks'][train_set_len + i]]], dtype=torch.float32)
            output = model(x)
            cost = criterion(output, y).float()
            test_loss += cost

        avg_test_cost = test_loss / test_set_len
    
    print('[Epoch: {:>4}] train cost = {:>.9}   test cost = {:>.9}'.format(epoch + 1, avg_train_cost, avg_test_cost))
    evaluation_result['test'].append(avg_test_cost.item())
    evaluation_result['train'].append(avg_train_cost.item())
    
plt.plot(range(training_epochs), evaluation_result['train'], evaluation_result['test'])
plt.legend(('train loss', 'test loss'))
plt.show()