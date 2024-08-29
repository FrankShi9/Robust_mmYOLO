import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
#from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

def load_label(file_path='label.txt'):
    bb = []
    with open(file_path, 'r') as f:
        for line in f:
            b = line.strip().split()[1:]  # Extracting vertex coordinates
            b = list(map(float, b))  # Converting coordinates to floats
            bb.append(b)
    return torch.tensor(bb)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class P4Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, num_classes):
        super(P4Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_dim, num_classes)
        self.rgbnet = models.resnet18(pretrained=True)
        self.rgbnet.fc = nn.Linear(in_features=self.rgbnet.fc.in_features, out_features=4)
        
    def forward(self, x, xr):
        rgbF = self.rgbnet(xr)
        x = x.permute(0,2,1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.fc(x)*rgbF
        return x
 
class PointDataset(Dataset):
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.x = [f for f in sorted(os.listdir(mat_path), key=len) if os.path.isfile(os.path.join(mat_path, f))] # x y match

    def __len__(self):
        if len(self.x)%bs:
            return len(self.x)-(len(self.x)%bs) # len round up
        else: 
            return len(self.x)
    
    def __getitem__(self, idx):
        num_indices = 80
        ridx = torch.randint(0, 400, (num_indices,))
        try:
            buf = scipy.io.loadmat(os.path.join(self.mat_path, self.x[-1]))['RawPoints'][:, :]
            buf = torch.tensor(buf)[ridx,:]
            x = buf
            y = load_label()[idx]

            return x, y
        except Exception as e:
            print(e)
            return None

def custom_collate_fn(batch):
    # Define your condition for skipping the batch
    if batch is None:
      skip_condition = True
    elif any(item is None for item in batch):
      skip_condition = True
    else:
      skip_condition = False
    
    if skip_condition:
        return None  # Return None or some placeholder
    
    inputs, targets = zip(*batch)
    
    # Stack or process the components as needed
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    return inputs, targets


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)-1

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform1 = transforms.Compose([
    transforms.ToTensor(),
])

def sync_dataloaders(loader1, loader2, loader3):
    for (batch1, batch2, batch3) in tqdm(zip(loader1, loader2, loader3), total=len(loader1)):
        yield batch1, batch2, batch3

if __name__ == '__main__':
  import warnings
  warnings.filterwarnings("ignore") #FIXME

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  global bs
  bs = 16 # input shape [bn, 3]
  epochs = 1
  persons = 1
  actions = 50

    
  model = P4Transformer(80, 80, 8, 128, 8, 4)
  model.load_state_dict(torch.load('checkpoint.pth'))
  model.to(device)
  model.eval()
  
  resnet = models.resnet18(pretrained=True)
  resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=10 * 2)
  resnet.load_state_dict(torch.load('joint-2.pth'))
  resnet.to(device)
  resnet.eval()

  loss = nn.L1Loss()

  prev_ls = 1e+2

  for e in range(epochs):
      print('epoch: '+str(e+1))
      total_ls = 0.0
      v_ls = 0.0
      for p in range(12,13): # p13
        print('person: ', str(p+1))
        flag = False
        for a in range(1):
            print('action: ', str(a+1))
            
            # train
            mm_path = "mmwave point cloud path"

            rgb_dir = "rgb image path"

            dataset = PointDataset(mm_path)
            dataloader = DataLoader(dataset, batch_size=bs, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)

            dataset = CustomDataset(rgb_dir, transform=transform)
            dataset1 = CustomDataset(rgb_dir, transform=transform1)

            test_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            infer_loader = DataLoader(dataset1, batch_size=bs, shuffle=False)

            total_ls = 0.0
            
            with torch.no_grad():
                for (batch1, batch2, batch3) in sync_dataloaders(test_loader, infer_loader, dataloader):
                    pred = None

                    x = batch1
                    x = x.to(device)
                    pred = resnet(x)
                    pred = pred.reshape(pred.shape[0], 10, 2)
                    pred = pred.to(torch.long)
                
                    xr = None

                    x = batch2
                    x = x.to(device)

                    for i in range(10):
                        for j in range(bs):
                            try:
                                x[j,:,pred[0,i,1]-30:pred[0,i,1]+30,pred[0,i,0]-30:pred[0,i,0]+30] = x[j,:,pred[0,i,1]+10:pred[0,i,1]+70,pred[0,i,0]-90:pred[0,i,0]-30] # y, x

                                new_size = (224, 224)
                                xr = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False) # input pretrain align
                            except:
                                pass

                    model.train()
        
                    if batch3 is None:
                        print("Skipping batch")
                        continue
        
                    x, y = batch3
                    
                    x, y = x.to(device), y.to(device)

                    x = x*torch.rand_like(x)
                    xr = xr*torch.rand_like(xr)
                    
                    output = model(x, xr)
                    
                    ls = loss(output, y)
                    total_ls += ls.item()

            print('loss:', total_ls)


    
