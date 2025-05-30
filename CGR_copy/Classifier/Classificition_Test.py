import torch
import CLIP.clip as clip
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
from torch import nn
def qprint(var,str):
    print("\033[92m"+"{}:{}".format(str,var)+"\033[0m")
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(512,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.model(x)
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"



model=Model().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001)

filepath='/home/yangxiaoda/CGR/Classifier/checkpoint/ck1.pth'
checkpoint = torch.load(filepath)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

clipmodel,preprocess=clip.load("ViT-B/32",device=device)
text=clip.tokenize(["Please help me find a similar dog."]).to(device)
with torch.no_grad():
    text_features=clipmodel.encode_text(text)

target=[0]
X=torch.tensor(text_features,dtype=torch.float32).to(device)
y=torch.tensor(target,dtype=torch.float32).view(-1,1).to(device)
train_dataset=TensorDataset(X,y)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=False)


model.eval()
with torch.no_grad():
    outputs=model(X)
    # qprint(outputs,'outputs')
    predicted=(outputs>0.5).float()
    if predicted==0:
        print("content")
    else:
        print("style")
    y=y.cpu()
    predicted=predicted.cpu()
    accuracy=accuracy_score(y,predicted)