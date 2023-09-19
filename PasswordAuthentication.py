import torch
import torch.nn as nn
import torch.optim as optim
# import keyboard

def getBatches(lst):
    big = []
    batch = []
    for lineCounter in range(0, len(lst)):
        if lines[lineCounter] != "\n":
            batch.append(lst[lineCounter])
        else:
            big.append(batch)
            batch = []
    return big



fData = open("Tanmay_keyTimings.txt", "r")
lines = fData.readlines()
fData.close()

fTargets = open("Tanmay_keyTargets.txt", "r")
targets = fTargets.readlines()
fTargets.close()

lines[0] = lines[0].strip("\n")
lines = [float(line.strip("\n")) if line!="\n" else line for line in lines]
lines.append("\n")

targets = [float(target.strip("\n")) for target in targets if target!="\n"]

data = getBatches(lines)

badIndices = [i for i in range(0, len(data)) if len(data[i])!=15]

for replace in range(0, len(data)):
    if replace in badIndices:
        data[replace] = None
        targets[replace] = None
cleaned_data = [datum for datum in data if datum!=None]
cleaned_targets = [target for target in targets if target!=None]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # super initializes a class of the superclass nn.Module

        self.hidden_layer = nn.Linear(15, 10) # input nodes = 15, output = 10
        self.output_layer = nn.Linear(10, 1) # input = 10, output = 1

    def forward(self, x): # the forward pass through the network
        x = torch.sigmoid(self.hidden_layer(x)) # hidden layer with activation
        x = torch.sigmoid(self.output_layer(x)) # output layer with activation

        return x

# Setup and Network Training

input_data = torch.tensor(cleaned_data) 
output_data = torch.tensor(cleaned_targets) # 0s and 1s, corresponding to input lists

model = NeuralNetwork()

lr = .1
numEpochs = 1000

loss_fn = nn.MSELoss() # mean-squared error

optimizer = optim.SGD(model.parameters(), lr=lr)

model.train()

for epoch in range(0, numEpochs):
    for i in range(0, input_data.shape[0]):
        prediction = model(input_data[i])
        error = loss_fn(output_data[i], prediction)

        # update the weights
        error.backward()
        optimizer.step()
        optimizer.zero_grad()


print ("Expected:")
print (output_data)
print ("Network output:")
print (model(input_data))

# print("Enter password:")
# for r in range(0, 1):
#     print(r)
#     recorded = keyboard.record(until="enter")
#     lst1=[]
#     vLines = []
#     for x in recorded:
        
#         lst1.append(x.time)

#     print(lst1)
#     print(len(lst1))
#     for i in range(1, len(lst1)):
#         print(i, lst1[i]-lst1[i-1])
#         vLines.append(lst1[i]-lst1[i-1])
#         # file.write(str(lst1[i]-lst1[i-1])+"\n")

#     keyboard.unhook_all()

torch.save(model, "modelSave.pt")

validationFile = open("validationData.txt", "r")
vLines = validationFile.readlines()
vLines = [float(line.strip("\n")) for line in vLines if line!="\n"]
x = model(torch.tensor(vLines))
print(x)
if x >= .75:
    print("Access Granted")
else:
    print("Access Denied")

