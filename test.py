import numpy as np
import torch
x=np.random.uniform(-1,1,100)
y=x*3+5+np.random.uniform(-0.01,0.01,100)
x=torch.from_numpy(x).view(100,1)
y=torch.from_numpy(y).view(100,1)
model=torch.nn.modules.Sequential(
    torch.nn.Linear(1,1)
)
opt=torch.optim.SGD(model.parameters(),lr=0.001)
def lossfn(train,test):
    return sum((train-test)**2)
loss_function = torch.nn.MSELoss()
l=100
for i in range(100):
    opt.zero_grad()
    input=x.float()
    output = model(input)
    print(model.state_dict())
    loss =lossfn(y.float(), output.float())
    print(loss)
    loss.backward()
    opt.step()