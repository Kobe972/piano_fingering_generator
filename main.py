from utils import *
from RL import *

file_path = '铁血丹心.mxl'
notes,score = get_notes_from_xml(file_path)
pure_notes=[] #处理一声部
for note in notes[0]:
    pure_notes.append(note[0])
fingering_part1=generate_fingering(pure_notes)
for i in range(len(fingering_part1)):
    for finger in fingering_part1[i]:
        addFingering(notes[0][i][1],finger)

pure_notes=[] #处理二声部
for note in notes[1]:
    pure_notes.append(note[1])
fingering_part2=generate_fingering(pure_notes)
for i in range(len(fingering_part2)):
    for finger in fingering_part2[i]:
        addFingering(notes[1][i][1],finger)

score.write('musicxml', fp='with_fingering.mxl')
