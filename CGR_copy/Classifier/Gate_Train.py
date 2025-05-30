import CLIP.clip as clip
# from ..CLIP import *
import torch
from PIL import Image
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score

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
# model, preprocess = clip.load("ViT-B/32", device=device)
# text=clip.tokenize(["content","style","give me a picture refer to the content","give me a picture refer to the style","Returns an image with similar content to this image","Returns an image with similar style to this image"]).to(device)
# target=[0,1,0,1,0,1]#target:[4,1]
# qprint(target.shape,'target')
# with torch.no_grad():
#     text_features=model.encode_text(text)#text_features:[4,512]

totlist=[]
zerofile='/home/yangxiaoda/CGR/Classifier/0emb.txt'
with open(zerofile,'r') as f:
    lines=f.read().strip().split('\n')
    for line in lines:
        line=line.strip().split(' ')
        float_list=[float(item) for item in line]
        linefloat=torch.tensor(float_list)
        totlist.append(linefloat)
onefile='/home/yangxiaoda/CGR/Classifier/1emb.txt'
with open(onefile,'r') as f:
    lines=f.read().strip().split('\n')
    for line in lines:
        line=line.strip().split(' ')
        float_list=[float(item) for item in line]
        linefloat=torch.tensor(float_list)
        totlist.append(linefloat)
        
text_features=torch.stack(totlist)
print(text_features.shape,'text_features.shape')
    
target=[0]*100+[1]*100
print(len(target),'len(target)')

# exit(0)
X_train,X_test,y_train,y_test=train_test_split(text_features,target,test_size=0.2,random_state=42)

X_train=torch.tensor(X_train,dtype=torch.float32).to(device)
y_train=torch.tensor(y_train,dtype=torch.float32).view(-1,1).to(device)
X_test=torch.tensor(X_test,dtype=torch.float32).to(device)
y_test=torch.tensor(y_test,dtype=torch.float32).view(-1,1).to(device)
train_dataset=TensorDataset(X_train,y_train)
train_loader=DataLoader(train_dataset,batch_size=2,shuffle=False)
test_dataset=TensorDataset(X_train,y_train)
test_loader=DataLoader(test_dataset,batch_size=2,shuffle=False)

model=Model().to(device)
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

qprint(X_train.shape,'train_dataset.shape')
qprint(y_train.shape,'y_train.shape')
qprint(X_test.shape,'X_test.shape')
qprint(y_test.shape,'y_test.shape')

epochs=100
for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        outputs=model(X_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
    # if (epoch+1)%10==0:
    #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    outputs=model(X_test)
    qprint(outputs,'outputs')
    predicted=(outputs>0.5).float()
    y_test=y_test.cpu()
    predicted=predicted.cpu()
    accuracy=accuracy_score(y_test,predicted)
    
print(f"Accuracy:{accuracy}")

filepath='/home/yangxiaoda/CGR/Classifier/checkpoint/ck1.pth'
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(checkpoint, filepath)





