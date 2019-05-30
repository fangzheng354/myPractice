# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
#import torch.nn.Parameter

n_data = torch.ones(100, 2)  # 生成一个100行2列的全1矩阵
x0 = torch.normal(2 * n_data, 1)  # 利用100行两列的全1矩阵产生一个正态分布的矩阵均值和方差分别是(2*n_data,1)
#y0 = -1*torch.ones(100,1)  # 给x0标定标签确定其分类0
y0 = -1*torch.ones(100,1)  # 给x0标定标签确定其分类0
 
x1 = torch.normal(-2 * n_data, 1)  # 利用同样的方法产生第二个数据类别
y1 = torch.ones(100,1)  # 但是x1数据类别的label就标定为1
 
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # cat方法就是将两个数据样本聚合在一起(x0,x1),0这个属性就是第几个维度进行聚合
y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # y也是一样
 
x = Variable(x)  # 将它们装载到Variable的容器里
y = Variable(y)  # 将它们装载到Variable的容器里


#dtype=torch.long, device=device


class textsvm(nn.Module):
    def __init__(self, dim, cla):
        super(textsvm, self).__init__()
        #self.args = args
        
        #dim = args.dim ## 已知词的数量
        #Dim = args.embed_dim ##每个词向量长度
        #Cla = args.class_num ##类别数
        #Ci = 1 ##输入的channel数
        #Knum = args.kernel_num ## 每种卷积核的数量
        #Ks = args.kernel_sizes ## 卷积核list，形如[2,3,4]

        self.w=nn.Parameter(Variable(torch.rand(dim,cla),requires_grad=True))
        self.b=nn.Parameter(Variable(torch.rand(1),requires_grad=True))
        #self.myparameters = nn.ParameterList(self.w, self.b)
        #self.convs = nn.ModuleList([nn.Conv2d(Ci,Knum,(K,Dim)) for K in Ks]) ## 卷积层
        #self.dropout = nn.Dropout(args.dropout) 
        #self.fc = nn.Linear(len(Ks)*Knum,Cla) ##全连接层
        
    def forward(self,x):
        y_pred= torch.matmul(x,self.w)+self.b
        #x = self.embed(x) #(N,W,D)
        
        #x = x.unsqueeze(1) #(N,Ci,W,D)
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
        #x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        
        #x = torch.cat(x,1) #(N,Knum*len(Ks))
        
        #x = self.dropout(x)
        #logit = self.fc(x)
        return y_pred
#net=textsvm(dim=2,cla=1)
#infile=open('','')
#self.myparameters = nn.ParameterList(Parameter1, Parameter2, ...)

#optimizer= torch.optim.SGD(textsvm,lr=0.0002)
net =  nn.Linear(2, 1)
#optimizer= torch.optim.Adam(net.parameters(),lr=0.02)
optimizer= torch.optim.SGD(net.parameters(),lr=0.02)


class MySVMLoss(nn.Module):
    def __init__(self):
        super(MySVMLoss, self).__init__()
        self.myloss2= torch.nn.MSELoss()
        self.reduction="mean"
        #print '1'
    def forward(self, output,y,svmmodel):
        #return  torch.mean(torch.mean((pred-truth)**2,1),0)
        loss = torch.mean(torch.clamp(1 - output * y.float(), min=0))  # hinge loss
        #loss += 0.1 * torch.mean(svmmodel.weight.t() ** 2)  # l2 penalty
        #loss_0 = torch.mean((output - y)**2)  # hinge loss
        #loss = torch.mean((output - y.float())**2)  # hinge loss
        #print "output .t-y shape"
        #print (output.t()-y.float()).size()
        #print y.size()
        #loss = torch.sum(torch.pow((output.t() - y.float()),2),0).squeeze()[0]# /y.size()[0]  # hinge loss
        #loss = self.myloss2(output,y)# /y.size()[0]  # hinge loss

        #loss=F.mse_loss(output, y,reduction="mean")
        #ret = (output - y) ** 2
        #loss = torch.mean(ret)
        #return [loss_0,loss]
        return loss



def myloss(output,y,svmmodel):
    loss = torch.mean(torch.clamp(1 - output.t() * y.float(), min=0))  # hinge loss
    loss += 0 * torch.mean(svmmodel.weight.t() ** 2)  # l2 penalty
    return loss

#loss_func = torch.nn.SoftMarginLoss()#
#loss_func = torch.nn.L1Loss()#
loss_func = torch.nn.MSELoss()#
#net.train()
print('training...')


criterion = MySVMLoss()
for t in range(1000):
    out = net(x)#100次迭代输出
    #criterion = MySVMLoss()
    #loss = loss_func(out,y.float())#计算loss为out和y的差异
    loss = criterion(out,y.float(),net)#计算loss为out和y的差异
    optimizer.zero_grad()#清除一下上次梯度计算的数值
    loss.backward()#进行反向传播

    #loss = torch.mean(torch.clamp(1 - out.t() * y.float(), min=0))  # hinge loss
    #loss += 5 * torch.mean(net.w ** 2)  # l2 penalty
    #loss.backward()


    optimizer.step()#最优化迭代
    if t %200 ==0:
        #prediction = torch.max(out,1)[1] ##返回每一行中最大值的那个元素，且返回其索引  torch.max()[1]， 只返回最大值的每个索引
        pred_y = out.data.numpy().squeeze()
        pred_y2=pred_y
        pred_y2[pred_y>0]=1
        pred_y2[pred_y<=0]=-1
        target_y = y.data.numpy().squeeze()
        accuracy = float((pred_y2 == target_y).astype(int).sum())/float(target_y.size)
        print "acc is "+str(accuracy)
        print net.weight.detach().numpy()
        print "w grad is "
        print net.weight.grad.detach().numpy()
        print "bias"
        print net.bias.detach().numpy()
        #print pred_y2
        print "loss "
        print loss.detach().numpy()
        #print loss_err.detach().numpy()
