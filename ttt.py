

# 环境 
# conda activate python39

# 尝试下高维张量的乘法

import torch

class ttt():
    # batch_size 就是同时做几道题。每道题都是一个棋谱。

    def __init__(self):
        width = 3**2
        hight = 1
        self.width = width
        self.hight = hight
        self.plate = torch.zeros(width,hight).cuda()

    def is_terminate(self):
        print('is_terminate')
        # print(self.plate)
        p = self.plate
        p = p.reshape(3,3)
        zr = torch.zeros(3).cuda()
        # print(p[0]==0)
        # print(zr,p)

        aa = 1
        bb = 2
        # 从玩家1 到玩家 2  。 0 表示 空地

        winner = 0
        for i in range(aa,bb+1):
            print('考察玩家',i)
            # s = p-i
            # print(s)

            # 检查 横向
            hz = torch.zeros(3).cuda()
            hz = hz+i
            print(hz)
            # 遍历每行
            for j in range(3):
                
                rs =torch.equal(p[j],hz) 
                if(rs ==1):
                    winner = i
                    break
            
            print("检查 每列")
            s = p.transpose(0,1)
            print('转置后',s)
            for j in range(3):    
                rs =torch.equal(p[j],hz) 
                if(rs ==1):
                    winner = i
                    break
            

            
            
            if(winner!=0):
                print('game over,winner is',winner)
                break
            else:
                print('no winner')
        return winner
        # print(torch.equal(p[0],zr))



class mlp():

    def __init__(self):
        super_param = [2,2] # 超参是个列表，记录有几层神经元，每层有几个神经元，列表里的值 不是个数 ，而是个数的对数。具体构造规则见build函数。
        self.super_param = super_param
    

    def set_super_param(self,super_param):
        self.set_super_param = super_param


    def build_nn(self):
        super_param=self.super_param


        depth = len(super_param)
        w_list = list()
        b_list = list()
        for i,ele in enumerate(super_param):
            if(i<=depth -2):
                kn = super_param[i]
                km = super_param[i+1]

                n = 2**kn
                m = 2**km



                if(self.mode == 'cpu'):
                    
                    w = torch.normal(0,1,(m,n)).cpu()
                    b = torch.normal(0,1,(m,self.batch_size)).cpu()
                elif(self.mode == 'gpu'):
                    w = torch.normal(0,1,(m,n)).cuda()
                    b = torch.normal(0,1,(m,self.batch_size)).cuda()

                    
                
                # 默认记录计算图
                w.requires_grad=True
                b.requires_grad=True


                w_list.append(w)
                b_list.append(b)
                    

        param = dict()
        param['w_list'] = w_list
        param['b_list'] = b_list
        param['depth'] = depth

        return param


    def forward(self,x,param):
        # y = 0


        w_list= param['w_list']
        b_list= param['b_list']

        # print('forward')

        depth = param['depth']
        for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.rl(w @ x + b)

        y = x
        return y


    def update(self,param):
        
        w_list= param['w_list']
        b_list= param['b_list']

        batch_size = self.batch_size
        lr = self.lr
        # print('update')

        with torch.no_grad():
            
            depth = param['depth']
            for i in range(depth-1): 
                # print('update',i)
                w = w_list[i]
                b = b_list[i]

                w -= lr * w.grad / batch_size
                w.grad.zero_()


                b -= lr * b.grad / batch_size
                b.grad.zero_()

class agent():

    def evaluate(self,tta):
        print(tta.plate)


        value = torch.normal(0,1,(tta.width,tta.hight))
        return value

a = ttt()
# a.is_terminate()

b = agent()
v = b.evaluate(a)
print(v)

# c = a.plate.reshape((3,3))
# print(a.plate)
# print(c)

pp = torch.tensor([1,1,1,2,0,2,0,2,0]).cuda()
pp = torch.tensor([1,0,1,1,0,2,1,2,0]).cuda()
a.plate = pp

a.is_terminate()