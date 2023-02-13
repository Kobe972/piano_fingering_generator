from music21 import *
from music21.musicxml import *

def name_to_num(name):
    if len(name.split('#'))>1:
        result=0.5
        result+=7*(int(name.split('#')[-1])-1)
        if name[0]=='D':
            result+=1
        elif name[0]=='E':
            result+=2
        elif name[0]=='F':
            result+=3
        elif name[0]=='G':
            result+=4
        elif name[0]=='A':
            result+=5
        elif name[0]=='B':
            result+=6
        return result
    elif len(name.split('-'))>1:
        result=-0.5
        result+=7*(int(name.split('-')[-1])-1)
        if name[0]=='D':
            result+=1
        elif name[0]=='E':
            result+=2
        elif name[0]=='F':
            result+=3
        elif name[0]=='G':
            result+=4
        elif name[0]=='A':
            result+=5
        elif name[0]=='B':
            result+=6
        return result
    else:
        result=0
        result+=7*(int(name[1:])-1)
        if name[0]=='D':
            result+=1
        elif name[0]=='E':
            result+=2
        elif name[0]=='F':
            result+=3
        elif name[0]=='G':
            result+=4
        elif name[0]=='A':
            result+=5
        elif name[0]=='B':
            result+=6
        return float(result)
    
def get_notes_from_xml(file_path):
    '''
    notes的元素个数等于声部数
    以第一声部为例，notes[0]含有两个元素，第一个元素包含每个音符的唱名，
    为一个列表（以处理和弦情况），第二个元素指代原始乐谱中的对应音符，用于
    向原始乐谱添加指法。
    score为原始乐谱，主要用于保存乐谱
    '''
    score = converter.parse(file_path)
    notes=[]
    for part in score.getElementsByClass(stream.Part):
        notes_part = []
        for element in part.flat.notes:
            if element.isChord:
                chord_notes = [name_to_num(note.nameWithOctave) for note in element]
                notes_part.append([chord_notes, element])
            else:
                notes_part.append([[name_to_num(element.nameWithOctave)], element])
        notes.append(notes_part)
    return notes,score

def addFingering(note,fingering):
    note.articulations.append(articulations.Fingering(fingering))

'''
以下是测试部分
解析音符，添加指法
保存文件
'''
# 读取 XML 文件并获取音符信息
file_path = '铁血丹心.mxl'
notes,score = get_notes_from_xml(file_path)
for i in range(len(notes[0])): #一声部
    # 为音符添加文本标注
    addFingering(notes[0][i][1],1)
    addFingering(notes[0][i][1],2)
    addFingering(notes[0][i][1],3)
    addFingering(notes[0][i][1],4)
for i in range(len(notes[1])): #二声部
    # 为音符添加文本标注
    addFingering(notes[1][i][1],1)
    addFingering(notes[1][i][1],2)
    addFingering(notes[1][i][1],3)
    addFingering(notes[1][i][1],4)

score.write('musicxml', fp='with_fingering.mxl')
