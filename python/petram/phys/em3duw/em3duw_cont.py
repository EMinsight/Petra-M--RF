'''
   continuity bdry: (= do nothing)
'''
from petram.phys.em3duw.em3duw_base import EM3DUW_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUW_cont')

def bdry_constraints():
    return [EM3DUW_Continuity]

class EM3DUW_Continuity(EM3DUW_Bdry):
    pass
