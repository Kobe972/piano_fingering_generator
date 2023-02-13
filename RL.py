import numpy as np

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
    result+=np.sum(np.abs(np.array(state)-np.array(action)))*2.5
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
        
    
