from utils import *
from brute import *

file_path = '铁血丹心.mxl'
notes,score = get_notes_from_xml(file_path)
print('生成一声部')
pure_notes=[] #处理一声部
for note in notes[0]:
    pure_notes.append(note[0])
fingering_part1=generate_fingering(pure_notes,0)
for i in range(len(fingering_part1)):
    for finger in fingering_part1[i]:
        addFingering(notes[0][i][1],finger)

print('生成二声部')
pure_notes=[] #处理二声部
for note in notes[1]:
    pure_notes.append(note[0])
fingering_part2=generate_fingering(pure_notes,1)
for i in range(len(fingering_part2)):
    for finger in fingering_part2[i]:
        addFingering(notes[1][i][1],6-finger)

score.write('musicxml', fp='with_fingering.mxl')
