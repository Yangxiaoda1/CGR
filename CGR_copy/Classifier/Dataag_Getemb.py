import CLIP.clip as clip
import torch
input_file = '/home/yangxiaoda/CGR/Classifier/0.txt'
with open(input_file,'r') as f:
    lines=f.read().strip().split('\n')

device="cuda" if torch.cuda.is_available() else "cpu"
model,preprocess= clip.load("ViT-B/32",device=device)

linelist=[]
for line in lines:
    linelist.append(line)
text=clip.tokenize(linelist).to(device)
with torch.no_grad():
    text_features=model.encode_text(text)

output_file = '/home/yangxiaoda/CGR/Classifier/0emb.txt'
with open(output_file,'w') as f:
    for line in text_features:
        line=line.tolist()
        line_str=' '.join(map(str,line))
        f.write(line_str+'\n')
        
        
        
        
input_file = '/home/yangxiaoda/CGR/Classifier/1.txt'
with open(input_file,'r') as f:
    lines=f.read().strip().split('\n')

device="cuda" if torch.cuda.is_available() else "cpu"
model,preprocess= clip.load("ViT-B/32",device=device)

linelist=[]
for line in lines:
    linelist.append(line)
text=clip.tokenize(linelist).to(device)
with torch.no_grad():
    text_features=model.encode_text(text)

output_file = '/home/yangxiaoda/CGR/Classifier/1emb.txt'
with open(output_file,'w') as f:
    for line in text_features:
        line=line.tolist()
        line_str=' '.join(map(str,line))
        f.write(line_str+'\n')