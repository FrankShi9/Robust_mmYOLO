import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio

fx = 375.66860062  # Focal length in x axis
fy = 375.66347079  # Focal length in y axis
cx = 319.99508973  # Optical center x
cy = 239.41364796  # Optical center y

mmp = "mmwave path"
rgbp = "rgb image path"

R = np.array(
    [[ 9.99458797e-01, 3.28646073e-02, 1.42475954e-03], 
     [4.78233954e-04, 2.87906567e-02, -9.99585349e-01], 
     [-3.28919997e-02, 9.99045052e-01, 2.87593582e-02]])

t = np.array(
    [-0.03981857,1.35834002,-0.05225502])

K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

Trc = np.row_stack((np.column_stack((R, t.T)),np.zeros((1,4))))

#rpc = np.load("pc-at.npy")

mat_contents = sio.loadmat(mmp)
ridx = np.random.randint(0, 400, (80,))
mmwave_data = mat_contents['RawPoints'][ridx,0:3].T
rcs = mat_contents['RawPoints'][ridx,3]
af = list(rcs)
colors = np.zeros((len(af), 4))
colors[:, 0] = 1  
af = af / np.linalg.norm(af)
colors[:, 3] = af

rpc = np.array(mmwave_data)
rpc = np.row_stack((rpc, np.ones((1,rpc.shape[1]))))

res = np.matmul(Trc, rpc)

x = res[0,:]
y = res[1,:]
z = res[2,:]

u = (fx * x) / z + cx
v = (fy * y) / z + cy

# print(u.shape, v.shape)
u,v = u-min(u)/(max(u)-min(u)), v-min(v)/(max(v)-min(v))

fig, ax = plt.subplots()
ax.imshow(Image.open(rgbp))
#ax.plot(u,v,'r.')
ax.scatter(u,v,c=colors)
fig.savefig('fig.png')
plt.show()
