from thop import profile
import torch
from ad import mmMesh

if __name__ == "__main__":
    bs = 16
    model = mmMesh()
    model.load_state_dict(torch.load('checkpoint-y.pth'))
    x, xr = torch.randn(bs, 80, 4), torch.randn(bs, 3, 480, 640)
    macs, params = profile(model, inputs=(x,xr))
    print(macs,params)
