
# coding: utf-8

# In[94]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# In[95]:


chainlength=30
max_no_of_atoms=10
max_no_of_sidechain_atoms=20
max_no_of_angles=3+max_no_of_sidechain_atoms
C=max_no_of_atoms+max_no_of_angles+20

res_indicator=np.array(np.zeros((20,chainlength))) #20 * chainlength
coordinates=np.array(np.random.rand(max_no_of_atoms,chainlength))#max_no_of_atoms * chainlength
angles=np.array(np.random.rand(max_no_of_angles,chainlength))#max_no_of_angles * chainlength
for i in range(0,chainlength):
    res_indicator[np.random.randint(0,20),i]=1

#output=np.random.rand(h,chainlength)
BATCH_SIZE=2
channels=[coordinates,angles,res_indicator]
channels=np.array(channels)
EPOCH=100

"""

conv1_out=
kernel_size=
stride=
padding=
conv2_out=
h=
pool1_size=
pool2_size="""
train_loader = Data.DataLoader(dataset=[channels, channels], batch_size=BATCH_SIZE, shuffle=True)
#test_loader=Data.DataLoader(dataset=,batch)


# In[96]:




cnn = CNN()
print(cnn)


# In[97]:



dcnn = DCNN()
print(dcnn)


# In[ ]:





# In[98]:





# In[99]:


model = VAE()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


# In[100]:


def loss_func(recon_theta,theta,mu,log_sigma):
    recons_loss=nn.MSELoss(recon_theta,theta)
    KL=0.5* torch.sum(exp(log_sigma)+ torch.square(mu)-1-log_sigma)
    return recons_loss+KL


# In[101]:


def train(epoch):
    model.train()
    train_loss=0
    
    for batch_id,data in enumerate(train_loader):
        data=Variable(torch.stack(data))
        optimizer.zero_grad()
        recons_theta,mu,log_sigma=model(data)
        loss=loss_func(recons_theta.view(chainlength,-1),data[coordinates:coordinates+angles],mu,log_sigma)
        loss.backward()
        train_loss+=loss
        optimizer.step()
        if batch_id % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_loader.dataset),
            100. * batch_id / len(train_loader),
            loss.data[0] / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


# In[102]:


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data)
        recons_theta, mu, log_sigma = model(data)
        test_loss += loss_func(recons_theta.view(chainlength,-1),data[coordinates:coordinates+angles],mu,log_sigma)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# In[103]:


#for epoch in range(1, EPOCH + 1):
 #   train(epoch)
   # test(epoch)
    


# In[104]:


a=[[1,2,3],[4,5,6],[7,8,9]]
b=[[10,11,12],[13,14,15]]
c=[16,17,18]
e=np.array(np.random.rand(2,3))
f=e+b
g=np.array(np.random.rand(3,3))
z=np.array(np.zeros((1,3)))
d=[f,g,z]
t=[d,d]
t=np.array(t)
print(t)
for batch,data in enumerate(t):
    print(batch,data)

