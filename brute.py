import numpy as np
import random
from tqdm import tqdm
from itertools import product
import copy

epochs=100
eps=0.2
batch_size=100
gamma=1
C=5
Q_updating_epochs=100
N=5

fixed=None #全局变量，指示哪些部分的指法被固定不可修改

def difficulty_single_syll(fingering,part):
    #此函数评价单个音符的指法所用
    result=0
    first=0
    second=1
    limitation_part0=np.array([[0,5,6,7,9],
                         [5,0,2,3,5],
                         [6,2,0,1,2],
                         [7,3,1,0,1],
                         [9,5,2,1,0]])
    limitation_part1=np.array([[0,1,2,5,9],
                         [1,0,1,3,7],
                         [2,1,0,2,6],
                         [5,3,2,0,5],
                         [9,7,6,5,0]])
    while second<len(fingering):
        if fingering[first]==0:
            first+=1
            second=max(second,first+1)
            continue
        if fingering[second]==0:
            second+=1
            continue
        result+=abs(fingering[second]-fingering[first]-(second-first))
        if part==0 and abs(fingering[second]-fingering[first])>limitation_part0[first][second] or part==1 and abs(fingering[second]-fingering[first])>limitation_part1[first][second]:
            result+=15 #和弦合理性判断
        first+=1
        second=max(second,first+1)
    return result

def difficulty_compound(fingering,part):
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
            if first==(0 if part==0 else 4): #穿指动作合法化
                if abs(fingering[second]-fingering[first]-(second-first))>0 and abs(fingering[second]-fingering[first]-(second-first))<5:
                    result+=2
                elif abs(fingering[second]-fingering[first]-(second-first))>=5:
                    result+=min(abs(fingering[second]-fingering[first]-(second-first)),15)
            elif abs(fingering[second]-fingering[first]-(second-first))>7:
                result+=15
                return result
            else:
                result+=2*abs(fingering[second]-fingering[first]-(second-first))
            first+=1
            second=max(second,first+1)
    return result

def penalty(state,action,part): #计算状态state采取步骤action后的惩罚,part是声部(0或1)
    result=difficulty_single_syll(action,part) #单独音节的惩罚，如果不是和弦显然为0
    for i in range(len(state)):
        if action[i]==0:
            action[i]=state[i]
        if state[i]==0:
            state[i]=action[i]
    if np.max(np.abs(np.array(state)-np.array(action)))<=4:
        result+=difficulty_compound(action,part)
    result+=min(np.max(np.abs(np.array(state)-np.array(action)))*2.5,15)
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

def brute_search(notes,part): #TODO: 增加参数指示上一次生成的指法，如果第一次搜索则为None
    fingering=[]
    N2=min(N,len(notes))
    for i in tqdm(range(-1,len(notes)+1-N2,N2)):
        action_list=[generate_fingering_per_syll(notes[j]).tolist() for j in range(i+1,min(len(notes),i+N2+1))]
        # TODO: 如果action_list某位置fixed值是True，则将其改为已经搜索出的指法，即对该位动作进行固定
        minimum_penalty=10000
        comb=[[*tmp] for tmp in product(*action_list)]
        for action_chain in comb:
            state=np.array(fingering[i]) if i!=-1 else np.array([0.,0.,0.,0.,0.])
            _penalty=0
            for action in action_chain:
                _penalty+=penalty(copy.deepcopy(state),copy.deepcopy(action),part)
                state=action
            if _penalty<minimum_penalty:
                minimum_penalty_action=copy.deepcopy(action_chain)
                minimum_penalty=_penalty
        for j in range(len(minimum_penalty_action)):
            if len(fingering)<=i+j+1:
                fingering.append(minimum_penalty_action[j])
            else:
                fingering[i+j+1]=minimum_penalty_action[j]
    return fingering

'''
TO BE DONE
增加brute_search_reversed，支持反向扫描搜索
'''

def evaluate(fingering,part): #对整个指法排布进行评估
    state=[0,0,0,0,0]
    _penalty=0
    for i in range(0,len(fingering)):
        _penalty+=penalty(copy.deepcopy(state),copy.deepcopy(fingering[i]),part)
        state=fingering[i]
    return _penalty
def generate_fingering(notes,part):
    fingering = np.array(brute_search(notes,part))
    '''
    TO BE DONE
    epsilon从0.1递增到0.9：
        随机选取epsilon的指法，设置对应位置fixed值为True
        调用brute_search_reversed，遇到fixed为True则对应位置指法不再进行搜索
        fixed取反
        调用brute_search，遇到fixed为True则对应位置指法不再进行搜索
        if evaluate(new_fingering)<evaluate(fingering):
            fingering=new_fingering\
    '''
    result=[]
    for i in range(len(fingering)):
        result.append(np.where(fingering[i]!=0)[0]+1)
    return result

