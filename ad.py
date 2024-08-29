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
import timm


def load_label(file_path='label.txt'):
    bb = []
    with open(file_path, 'r') as f:
        for line in f:
            b = line.strip().split()[1:]  # Extracting vertex coordinates
            b = list(map(float, b))  # Converting coordinates to floats
            bb.append(b)
    return torch.tensor(bb)

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        # bs = 16 # Profile
        x = F.relu(self.bn1(self.conv1(x).T)) #FIXME
        x = F.relu(self.bn2(self.conv2(x.T).T))
        x = F.relu(self.bn3(self.conv3(x.T).T))
        #x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(3,1) #FIXME
        # if x.is_cuda:
        #     iden = iden.cuda()
        # print(x.shape, iden.shape)
        # x = x + iden
        
        #x = x.view(-1, 3)
        x = x.view(bs,3,80) # 2 GPU //2 else delete
        return x


class mmMesh(nn.Module):
    def __init__(self):
        super(mmMesh, self).__init__()
        # self.base = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3), nn.ReLU())
        self.base = PointNet()
        self.attention = nn.MultiheadAttention(embed_dim=80, num_heads=8, batch_first=True)
        self.globM = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.ReLU())
        self.globAttW = nn.Parameter(torch.randn(3, 1)) # rand init learnable
        self.globL = nn.LSTM(1, 3, 3, batch_first=True)
        self.sharedM = nn.Linear(188, 1) #FIXME
        self.dt = nn.Conv1d(3, 1, 1)
        self.at = nn.LSTM(1, 2, 3, batch_first=True)
        self.fuse = nn.Linear(11, 4)
        self.AP = nn.Linear(3, 36*3)
        self.softmax = nn.Softmax()
        self.sharedMW = nn.Parameter(torch.randn(3, 1))
        self.attS = nn.Linear(4,3)
        #self.rgbnet = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
        self.rgbnet = models.resnet18(pretrained=True)
        self.rgbnet.fc = nn.Linear(in_features=self.rgbnet.fc.in_features, out_features=4)
        #self.rgbnet.fc = nn.Linear(in_features=self.rgbnet.fc.in_features, out_features=11)
       

    def forward(self, x, xr):
        # bs=16 # Profile
        #xr = F.scaled_dot_product_attention(xr,xr,xr)
        #xr = F.interpolate(xr, size=(224,224))
        rgbF = self.rgbnet(xr)
        x = x.permute(0,2,1) # [bs,4,80]
        ax = self.attention(x,x,x)[0] # [bs,4,80]
        ax = ax.permute(0,2,1) # [bs,80,4]
        ax = self.attS(ax) # [bs,80,3]
        ax = ax.permute(0,2,1) # [bs,3,80]
        
        # print('x: ', x.shape, 'r: ', self.base(x).shape)
        anchorX = torch.add(ax,self.base(x))
        #anchorX = ax*self.base(x)
        #x = self.base(anchorX)
        globC = self.globM(self.base(x))
        # print('globC: ', globC.shape, 'globAttW: ', self.globAttW.shape)
        globF = self.softmax(self.globAttW*globC)

        #globF = globC # Profiling only

        #globF = F.scaled_dot_product_attention(globC, globC, globC)
        globG = self.globL(globF)[0]
        # print(self.AP(globG).shape)
        APG = self.AP(globG)
        # print(APG.shape, anchorX.shape)
        br = torch.cat([APG, anchorX], 2)
        sharedM = self.sharedM(br)*self.sharedMW
        dt = self.dt(sharedM)
        at = self.at(dt)[0].reshape(bs,2) # 2 GPU //2 else delete
        # print(globG.shape, at.shape)
        return self.fuse(torch.cat([globG.reshape(bs,9), at], 1))*rgbF # 2 GPU //2 else delete

        # buf = torch.cat([globG.reshape(bs,9), at], 1)*rgbF
        # fus = F.scaled_dot_product_attention(buf,buf,buf)
        # return self.fuse(fus)
 

class PointDataset(Dataset):
    def __init__(self, mat_path, val='label.txt'):
        self.mat_path = mat_path
        self.x = [f for f in sorted(os.listdir(mat_path), key=len) if os.path.isfile(os.path.join(mat_path, f))] # x y match
        self.val = val

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
            y = load_label(self.val)[idx]

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

  writer = SummaryWriter(log_dir='')

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  global bs
  bs = 16
  epochs = 15
  persons = 1
  actions = 50
  
  model = mmMesh()
  model.to(device)

  resnet = models.resnet18(pretrained=True)
  resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=10 * 2)
  resnet.load_state_dict(torch.load('joint.pth'))
  resnet.to(device)
  resnet.eval()


  # if torch.cuda.device_count() > 1:
  #   print(f"Let's use {torch.cuda.device_count()} GPUs!")
  #   model = nn.DataParallel(model)

  
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
  loss = nn.L1Loss()

  prev_ls = 1e+2

  for e in range(epochs):
      print('epoch: '+str(e+1))
      total_ls = 0.0
      v_ls = 0.0
      for p in range(persons):
        print('person: ', str(p+1))
        flag = False
        for a in range(actions):
            print('action: ', str(a+1))
            
            # train
            mm_path = "mmwave point cloud path"
            vmm_path = "validation mmwave point cloud path"

            rgb_dir = "rgb image path"
            vrgb_dir = "validation rgb image path"

            dataset = PointDataset(mm_path)
            dataloader = DataLoader(dataset, batch_size=bs, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)

            vdataset = PointDataset(vmm_path, val='label-val.txt')
            vdataloader = DataLoader(vdataset, batch_size=bs, num_workers=0, shuffle=True, collate_fn=custom_collate_fn)

            dataset = CustomDataset(rgb_dir, transform=transform)
            dataset1 = CustomDataset(rgb_dir, transform=transform1)
            vdataset = CustomDataset(vrgb_dir, transform=transform)
            vdataset1 = CustomDataset(vrgb_dir, transform=transform1)

            test_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            infer_loader = DataLoader(dataset1, batch_size=bs, shuffle=False)
            vtest_loader = DataLoader(vdataset, batch_size=bs, shuffle=False)
            vinfer_loader = DataLoader(vdataset1, batch_size=bs, shuffle=False)

            for (batch1, batch2, batch3) in sync_dataloaders(test_loader, infer_loader, dataloader):

                pred = None

                x = batch1
                x = x.to(device)
                pred = resnet(x)
                pred = pred.reshape(pred.shape[0], 10, 2)
                pred = pred.to(torch.long)
                #print(pred[0])
            
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
                #print('in', x.shape, rcs.shape)
    
                optimizer.zero_grad()
                
                x, y = x.to(device), y.to(device)
                #x = x*torch.rand_like(x)
                output = model(x, xr)
                
                ls = loss(output, y)
                
                ls.backward()
                optimizer.step()
    
                total_ls += ls.item()
                
                print('loss:', ls.item())

            writer.add_scalar('training_loss', total_ls, (e+1))

            model.eval()
            
            with torch.no_grad():
                for (batch1, batch2, batch3) in sync_dataloaders(vtest_loader, vinfer_loader, vdataloader):
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

        
                    if batch3 is None:
                        print("Skipping batch")
                        continue
        
                    x, y = batch3
                    
                    x, y = x.to(device), y.to(device)

                    output = model(x, xr)
                    
                    ls = loss(output, y)
                    v_ls += ls.item()
                    print('val-loss:', ls.item())

            writer.add_scalar('val_loss', v_ls, (e+1))

            #torch.cuda.empty_cache()

  #       print(total_ls/len(dataloader))

    #   if ls.item()<prev_ls:
    #       torch.save(model.state_dict(), './checkpoint-ya.pth')
    #       prev_ls = ls.item()

      #writer.add_scalar('training_loss', total_ls, (i+1))
      #writer.add_scalar('val_loss', v_ls, (i+1))

  writer.close()


    

