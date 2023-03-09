import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import copy

epochs=100
eps=0.2
batch_size=100
gamma=1
C=5
Q_updating_epochs=100

cache_capacity=0

def difficulty_single_syll(fingering):
    #此函数评价单个音符的指法所用
    result=0
    first=0
    second=1
    while second<len(fingering):
        if fingering[first]==0:
            first+=1
            second=max(second,first+1)
            continue
        if fingering[second]==0:
            second+=1
            continue
        result+=abs(fingering[second]-fingering[first]-(second-first))
        first+=1
        second=max(second,first+1)
    return result

def difficulty_compound(fingering):
    #用于评估组合后的指法
    result=0
    first=0
    second=1
    while second<len(fingering):
        if fingering[first]==0:
            first+=1
            second=max(second,first+1)
            continue
        if fingering[second]==0:
            second+=1
            continue
        if fingering[second]>fingering[first]: #正常相对顺序
            result+=abs(fingering[second]-fingering[first]-(second-first))
            first+=1
            second=max(second,first+1)
            continue
        else:
            if first==0: #穿指动作合法化
                result+=abs(fingering[second]-fingering[first]-(second-first))
            else:
                result+=2*abs(fingering[second]-fingering[first]-(second-first))
            first+=1
            second=max(second,first+1)
    return result

def penalty(state,action): #计算状态state采取步骤action后的惩罚
    result=difficulty_single_syll(action) #单独音节的惩罚，如果不是和弦显然为0
    state=state[:-1]
    for i in range(len(state)):
        if action[i]==0:
            action[i]=state[i]
        if state[i]==0:
            state[i]=action[i]
    result+=difficulty_compound(action)
    result+=np.max(np.abs(np.array(state)-np.array(action)))*2.5
    return float(result)

def choose_n_from_m(m, n):
    res = []
    curr = [-1] * n
    i = 0
    while i >= 0:
        curr[i] += 1
        if curr[i] > m-1:
            i -= 1
        elif i == n - 1:
            res.append(curr[:])
        else:
            i += 1
            curr[i] = curr[i - 1]
    return res

def generate_fingering_per_syll(syll):
    #得到单个音符的所有指法排布，即得到当前的动作集合
    num_notes=len(syll)
    idxs=choose_n_from_m(5,num_notes)
    result=np.zeros((len(idxs),5))
    for i in range(len(idxs)):
        ind=0
        for j in idxs[i]:
            result[i][j]=syll[ind]
            ind+=1
    return result

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(11,128),
            nn.Sigmoid(),
            nn.Linear(128,64),
            nn.Sigmoid(),
            nn.Linear(64,32),
            nn.Sigmoid(),
            nn.Linear(32,1))
    def forward(self,x):
        return self.layer(x)
    def train(self,x,y):
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,eps=1e-8,betas=(0.9,0.999))
        y=torch.tensor(y)
        x=torch.tensor(x)
        for i in range(Q_updating_epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
class LimitedList:
    def __init__(self, capacity):
        self.capacity = capacity
        self.list = []

    def append(self, item):
        if len(self.list) >= self.capacity:
            self.list.pop(0)
        self.list.append(item)

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)
def get_argmin_Q(Q,state,action_list):
    state_action=[] #每个元素是state和action_list的对应项合并的结果
    for action in action_list:
        state_action.append(state.tolist()+action.tolist())
    Q_values=Q(torch.tensor(state_action))
    idx=Q_values.argmin()
    action=action_list[idx]
    return action,Q(torch.tensor(state_action[idx]))
def get_trained_Q(notes):
    Q=DQN()
    Q_hat=DQN()
    D=[]
    for epoch in tqdm(range(0,epochs)):
        state=np.array([0,0,0,0,0,-1])
        for t in range(-1,len(notes)-1):
            action_list=generate_fingering_per_syll(notes[t+1])
            if random.random()<eps:
                a_t=action_list[random.randint(0,len(action_list)-1)]
            else:
                a_t,_=get_argmin_Q(Q,state,action_list)
            r_t=penalty(copy.deepcopy(state),copy.deepcopy(a_t))
            state_t_plus_1=np.concatenate((a_t,[state[-1]+1]))
            D.append((state,a_t,r_t,state_t_plus_1))
            _batch_size=min(batch_size,len(D))
            sampled=random.sample(D,_batch_size)
            x=[]
            y=[]
            for sample in sampled:
                _state,_a_t,_r_t,_next_state=sample
                if _state[-1]==len(notes)-1:
                    y.append([_r_t])
                else:
                    action_list=generate_fingering_per_syll(notes[int(_state[-1])+1])
                    argmin_action,argmin_Q=get_argmin_Q(Q_hat,_state,action_list)
                    x.append(np.concatenate((_state,argmin_action)).astype(np.float32))
                    y.append([_r_t+gamma*argmin_Q])
            x=np.array(x)
            Q.train(x,y)
            state=state_t_plus_1
        if (epoch+1)%C==0 or epoch==epochs-1:
            Q_hat.layer.load_state_dict(Q.layer.state_dict())
    return Q_hat
def generate_fingering(notes):
    fingering_list=[]
    Q=get_trained_Q(notes)
    state=np.array([0,0,0,0,0,-1])
    for t in range(-1,len(notes)-1):
        action_list=generate_fingering_per_syll(notes[t+1])
        a_t,_=get_argmin_Q(Q,state,action_list)
        state=np.array(a_t.tolist()+[t+1])
        fingering_list.append(np.where(a_t!=0)[0]+1)
    return fingering_list

