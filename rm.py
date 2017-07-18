#!/usr/bin/python

import numpy as np
import hoomd
import sys
import os
import collections

################################################################
# Creating Initial Stucture
################################################################

class OrderedSet(collections.MutableSet):
    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable
    def __len__(self):
        return len(self.map)
    def __contains__(self, key):
        return key in self.map
    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]
    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev
    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]
    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]
    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key
    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))
    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

def A(alist):
    return np.array(alist)

hoomd.context.initialize("--gpu=0 --ignore-display-gpu --minimize-cpu-usage")

atomToParticle = {
        # AOT #
        "S":     {"type":   "SL", "bonds": [A([0,1]),A([0,2]),A([0,3]),A([0,4])],
            "angles": [A([0,4,5]),A([0,4,9]),A([0,4,6])],
            "dihedrals": [A([0,4,9,10]),A([0,4,9,11]),A([0,4,6,22])]},
        "OS1":   {"type":  "O2L", "bonds": [],
            "angles": [A([0,-1,1]),A([0,-1,2]),A([0,-1,3])],
            "dihedrals": []},
        "OS2":   {"type":  "O2L", "bonds": [],
            "angles": [A([0,-2,1]),A([0,-2,2])],
            "dihedrals": []},
        "OS3":   {"type":  "O2L", "bonds": [],
            "angles": [A([0,-3,1])],
            "dihedrals": []},
        "C1":    {"type": "CTL1", "bonds": [A([0,1]),A([0,2]),A([0,5])],
            "angles": [A([0,2,3]),A([0,2,4]),A([0,2,18]),A([0,5,6]),A([0,5,7])],
            "dihedrals": [A([0,5,7,8]),A([0,2,18,19]),A([0,2,18,20])]},#5
        "H1":    {"type": "HAL1", "bonds": [],
            "angles": [A([0,-1,4]),A([0,-1,1])],
            "dihedrals": []},
        "C2":    {"type": "CTL2", "bonds": [A([0,1]),A([0,2]),A([0,16])],
            "angles": [A([0,16,17]),A([0,16,18])],
            "dihedrals": [A([0,-2,3,4]),A([0,-2,3,5]),A([0,16,18,19])]},
        "H2":    {"type": "HAL2", "bonds": [],
            "angles": [A([0,-1,1]),A([0,-1,15])],
            "dihedrals": []},
        "H3":    {"type": "HAL2", "bonds": [],
            "angles": [A([0,-2,14])],
            "dihedrals": []},
        "C3":    {"type":   "CL", "bonds": [A([0,1]),A([0,2])],
            "angles": [A([0,-5,-3]),A([0,2,3])],
            "dihedrals": [A([0,2,3,6]),A([0,-5,-3,13])]},#10
        "O1":    {"type":  "OBL", "bonds": [],
            "angles": [A([0,-1,1])],
            "dihedrals": [A([0,-1,1,2])]},
        "O2":    {"type":  "OSL", "bonds": [A([0,1])],
            "angles": [A([0,1,2]),A([0,1,3]),A([0,1,4])],
            "dihedrals": [A([0,1,4,5]),A([0,1,4,9])]},
        "C4":    {"type": "CTL2", "bonds": [A([0,1]),A([0,2]),A([0,3])],
            "angles": [A([0,3,4]),A([0,3,8])],
            "dihedrals": [A([0,3,4,5]),A([0,3,8,9])]},
        "H4":    {"type": "HAL2", "bonds": [],
            "angles": [A([0,-1,2])],
            "dihedrals": []},
        "H5":    {"type": "HAL2", "bonds": [],
            "angles": [A([0,-2,1])],
            "dihedrals": []},#15
        "C5":    {"type":   "CH", "bonds": [A([0,1]),A([0,5])],
            "angles": [A([0,1,2]),A([0,5,6])],
            "dihedrals": [A([0,1,2,3])]},
        "C6":    {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [A([0,-1,4]),A([0,1,2])],
            "dihedrals": [A([0,1,2,3]),A([0,-1,4,5])]},
        "C7":    {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [],
            "dihedrals": []},
        "C8":    {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [],
            "dihedrals": []},
        "C9":    {"type":  "C3a", "bonds": [],
            "angles": [],
            "dihedrals": []},#20
        "C10":   {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [],
            "dihedrals": [A([0,-5,-4,-3])]},
        "C11":   {"type":  "C3a", "bonds": [],
            "angles": [],
            "dihedrals": []},
        "C12":   {"type":   "CL", "bonds": [A([0,1]),A([0,2])],
            "angles": [A([0,2,3])],
            "dihedrals": [A([0,2,3,6])]},
        "O3":    {"type":  "OBL", "bonds": [],
            "angles": [A([0,-1,1])],
            "dihedrals": [A([0,-1,1,2])]},
        "O4":    {"type":  "OSL", "bonds": [A([0,1])],
            "angles": [A([0,1,2]),A([0,1,3]),A([0,1,4])],
            "dihedrals": [A([0,1,4,5]),A([0,1,4,9])]},#25
        "C13":   {"type": "CTL2", "bonds": [A([0,1]),A([0,2]),A([0,3])],
            "angles": [A([0,3,4]),A([0,3,8])],
            "dihedrals": [A([0,3,4,5]),A([0,3,8,9])]},
        "H21":   {"type": "HAL2", "bonds": [],
            "angles": [A([0,-1,1]),A([0,-1,2])],
            "dihedrals": []},
        "H22":   {"type": "HAL2", "bonds": [],
            "angles": [A([0,-2,1])],
            "dihedrals": []},
        "C14":   {"type":   "CH", "bonds": [A([0,1]),A([0,5])],
            "angles": [A([0,1,2]),A([0,5,6])],
            "dihedrals": [A([0,1,2,3])]},
        "C15":   {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [A([0,-1,4]),A([0,1,2])],
            "dihedrals": [A([0,1,2,3]),A([0,-1,4,5])]},#30
        "C16":   {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [A([0,1,2])],
            "dihedrals": []},
        "C17":   {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [],
            "dihedrals": []},
        "C18":   {"type":  "C3a", "bonds": [],
            "angles": [],
            "dihedrals": []},
        "C19":   {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [],
            "dihedrals": [A([0,-5,-4,-3])]},
        "C20":   {"type":  "C3a", "bonds": [],
            "angles": [],
            "dihedrals": []},#35
        "NA":    {"type":  "SOD", "bonds": [],
            "angles": [],
            "dihedrals": []},
        # WATER #
        "OW":    {"type":   "OT", "bonds": [A([0,1]),A([0,2])],
            "angles": [],
            "dihedrals": []},
        "HW1":   {"type":   "HT", "bonds": [A([0,1])],
            "angles": [A([0,-1,1])],
            "dihedrals": []},
        "HW2":   {"type":   "HT", "bonds": [],
            "angles": [],
            "dihedrals": []},
        # ISOOCTANE #
        "C21":   {"type":  "C3b", "bonds": [A([0,1])],
            "angles": [A([0,1,5])],
            "dihedrals": []},#1
        "C22":   {"type":    "C", "bonds": [A([0,1]),A([0,4]),A([0,5])],
            "angles": [A([0,1,2])],
            "dihedrals": [A([0,1,2,3])]},#2
        "C23":   {"type":  "CH2", "bonds": [A([0,1])],
            "angles": [A([0,-1,-2]),A([0,-1,4]),A([0,-1,3]),A([0,1,2]),A([0,1,5])],
            "dihedrals": []},#3
        "C24":   {"type":   "CH", "bonds": [A([0,1]),A([0,4])],
            "angles": [],
            "dihedrals": [A([0,-1,-2,2])]},#4
        "C25":   {"type":  "C3b", "bonds": [],
            "angles": [A([0,-1,3])],
            "dihedrals": []},#5
        "C26":   {"type":  "C3b", "bonds": [],
            "angles": [],
            "dihedrals": []},#6
        "C27":   {"type":  "C3b", "bonds": [],
            "angles": [A([0,-5,-1]),A([0,-5,-6])],
            "dihedrals": []},#7
        "C28":   {"type":  "C3b", "bonds": [],
            "angles": [],
            "dihedrals": []},#8
        # IONS #
        "CL":    {"type":  "CLS", "bonds": [],
            "angles": [],
            "dihedrals": []},
        "ZR":    {"type":  "ZRS", "bonds": [],
            "angles": [],
            "dihedrals": []}
        }

Particles = {
        "CTL1": {"id": 0, "mass": 12.0110, "charge":-0.1900}, 
        "CTL2": {"id": 1, "mass": 12.0110, "charge":-0.1800},
        "CTL3": {"id": 2, "mass": 12.0110, "charge":-0.2700},
        "SL":   {"id": 3, "mass": 32.0600, "charge": 1.3600}, 
        "O2L":  {"id": 4, "mass": 15.9994, "charge":-0.6000}, 
        "HAL1": {"id": 5, "mass":  1.0080, "charge": 0.0900}, 
        "HAL2": {"id": 6, "mass":  1.0080, "charge": 0.0900}, 
        "HAL3": {"id": 7, "mass":  1.0080, "charge": 0.0900},
        "CL":   {"id": 8, "mass": 12.0110, "charge": 0.6300}, 
        "OBL":  {"id": 9, "mass": 15.9994, "charge":-0.5200}, 
        "OSL":  {"id":10, "mass": 15.9994, "charge":-0.3400}, 
        "CT0":  {"id":11, "mass": 12.0110, "charge": 0.0000}, 
        "SOD":  {"id":12, "mass": 22.9898, "charge": 1.0000}, 
        "HT":   {"id":13, "mass":  1.0080, "charge": 0.4170},
        "OT":   {"id":14, "mass": 15.9994, "charge":-0.8340}, 
        "C3a":  {"id":15, "mass": 15.0350, "charge": 0.0000}, 
        "C3b":  {"id":16, "mass": 15.0350, "charge": 0.0000}, 
        "C3c":  {"id":17, "mass": 15.0350, "charge": 0.0000}, 
        "CH2":  {"id":18, "mass": 14.0270, "charge": 0.0000}, 
        "CH":   {"id":19, "mass": 13.0190, "charge": 0.0000},
        "C":    {"id":20, "mass": 12.0110, "charge": 0.0000}, 
        "ZRS":  {"id":21, "mass": 91.2240, "charge": 4.0000}, 
        "CLS":  {"id":22, "mass": 35.4530, "charge":-1.0000}
        }

Bonds = {
        "CTL3,CL":      {"id": 1, "func": 1, "b0": 0.15220, "kb": 167360.00},
        "CTL2,CL":      {"id": 2, "func": 1, "b0": 0.15220, "kb": 167360.00},
        "CTL1,CL":      {"id": 3, "func": 1, "b0": 0.15220, "kb": 167360.00},
        "OBL,CL":       {"id": 4, "func": 1, "b0": 0.12200, "kb": 627600.00},
        "OSL,CL":       {"id": 5, "func": 1, "b0": 0.13340, "kb": 125520.00},
        "CTL1,HAL1":    {"id": 6, "func": 1, "b0": 0.11110, "kb": 258571.20},
        "CTL2,HAL2":    {"id": 7, "func": 1, "b0": 0.11110, "kb": 258571.20},
        "CTL3,HAL3":    {"id": 8, "func": 1, "b0": 0.11110, "kb": 269449.60},
        "CTL3,OSL":     {"id": 9, "func": 1, "b0": 0.14300, "kb": 284512.00},
        "CTL2,OSL":     {"id":10, "func": 1, "b0": 0.14300, "kb": 284512.00},
        "CTL1,OSL":     {"id":11, "func": 1, "b0": 0.14300, "kb": 284512.00},
        "CTL1,CTL1":    {"id":12, "func": 1, "b0": 0.15000, "kb": 186188.00},
        "CTL1,CTL2":    {"id":13, "func": 1, "b0": 0.15380, "kb": 186188.00},
        "CH,CTL2":      {"id":14, "func": 1, "b0": 0.15380, "kb": 186188.00},
        "CTL1,CTL3":    {"id":15, "func": 1, "b0": 0.15380, "kb": 186188.00},
        "CTL2,CTL2":    {"id":16, "func": 1, "b0": 0.15300, "kb": 186188.00},
        "CTL2,CTL3":    {"id":17, "func": 1, "b0": 0.15280, "kb": 186188.00},
        "CTL3,CTL3":    {"id":18, "func": 1, "b0": 0.15300, "kb": 186188.00},
        "SL,O2L":       {"id":19, "func": 1, "b0": 0.14480, "kb": 451872.00},
        "SL,OSL":       {"id":20, "func": 1, "b0": 0.15750, "kb": 209200.00},
        "SL,CTL1":      {"id":21, "func": 1, "b0": 0.18000, "kb": 376560.00},
        "CT0,CTL3":     {"id":22, "func": 1, "b0": 0.15380, "kb": 186188.00},
        "CT0,CTL2":     {"id":23, "func": 1, "b0": 0.15380, "kb": 186188.00},
        "HT,HT":        {"id":24, "func": 1, "b0": 0.15139, "kb": 376560.00},
        "HT,OT":        {"id":25, "func": 1, "b0": 0.09572, "kb": 376560.00},
        "CH,CH2":       {"id":26, "func": 1, "b0": 0.15400, "kb": 80235.022},
        "CH2,C":        {"id":27, "func": 1, "b0": 0.15400, "kb": 80235.022},
        "C3b,C":        {"id":28, "func": 1, "b0": 0.15400, "kb": 80235.022},
        "C3b,CH":       {"id":29, "func": 1, "b0": 0.15400, "kb": 80235.022},
        "CH,C3c":       {"id":30, "func": 1, "b0": 0.15400, "kb": 80235.022},
        "CH2,CH2":      {"id":31, "func": 1, "b0": 0.15400, "kb": 80235.022},
        "CH2,C3a":      {"id":32, "func": 1, "b0": 0.15400, "kb": 80235.022}
        }

Angles = {
        "OBL,CL,CTL3":      {"id": 1, "func": 1, "theta": 125.00, "ktheta": 585.76000},
        "OBL,CL,CTL2":      {"id": 2, "func": 1, "theta": 125.00, "ktheta": 585.76000},
        "OBL,CL,CTL1":      {"id": 3, "func": 1, "theta": 125.00, "ktheta": 585.76000},
        "OSL,CL,OBL":       {"id": 4, "func": 1, "theta": 125.90, "ktheta": 753.12000},
        "CL,OSL,CTL1":      {"id": 5, "func": 1, "theta": 109.60, "ktheta": 334.72000},
        "CL,OSL,CTL2":      {"id": 6, "func": 1, "theta": 109.60, "ktheta": 334.72000},
        "CL,OSL,CTL3":      {"id": 7, "func": 1, "theta": 109.60, "ktheta": 334.72000},
        "HAL2,CTL2,CL":     {"id": 8, "func": 1, "theta": 109.50, "ktheta": 276.14400},
        "HAL3,CTL3,CL":     {"id": 9, "func": 1, "theta": 109.50, "ktheta": 276.14400},
        "CTL2,CTL2,CL":     {"id":10, "func": 1, "theta": 108.00, "ktheta": 435.13600},
        "CTL3,CTL2,CL":     {"id":11, "func": 1, "theta": 108.00, "ktheta": 435.13600},
        "CTL2,CTL1,CL":     {"id":12, "func": 1, "theta": 108.00, "ktheta": 435.13600},
        "CTL1,CTL2,CL":     {"id":13, "func": 1, "theta": 108.00, "ktheta": 435.13600},
        "OSL,CL,CTL3":      {"id":14, "func": 1, "theta": 109.00, "ktheta": 460.24000},
        "OSL,CL,CTL2":      {"id":15, "func": 1, "theta": 109.00, "ktheta": 460.24000},
        "OSL,CTL1,CTL2":    {"id":16, "func": 1, "theta": 110.10, "ktheta": 633.45760},
        "OSL,CTL1,CTL3":    {"id":17, "func": 1, "theta": 110.10, "ktheta": 633.45760},
        "OSL,CTL2,CTL1":    {"id":18, "func": 1, "theta": 110.10, "ktheta": 633.45760},
        "OSL,CTL2,CH":      {"id":19, "func": 1, "theta": 110.10, "ktheta": 633.45760},
        "OSL,CTL2,CTL2":    {"id":20, "func": 1, "theta": 110.10, "ktheta": 633.45760},
        "OSL,CTL2,CTL3":    {"id":21, "func": 1, "theta": 110.10, "ktheta": 633.45760},
        "HAL2,CTL2,HAL2":   {"id":22, "func": 1, "theta": 109.00, "ktheta": 297.06400},
        "HAL3,CTL3,HAL3":   {"id":23, "func": 1, "theta": 108.40, "ktheta": 297.06400},
        "HAL1,CTL1,OSL":    {"id":24, "func": 1, "theta": 109.50, "ktheta": 502.08000},
        "HAL2,CTL2,OSL":    {"id":25, "func": 1, "theta": 109.50, "ktheta": 502.08000},
        "HAL3,CTL3,OSL":    {"id":26, "func": 1, "theta": 109.50, "ktheta": 502.08000},
        "HAL1,CTL1,CTL1":   {"id":27, "func": 1, "theta": 110.10, "ktheta": 288.69600},
        "HAL1,CTL1,CTL2":   {"id":28, "func": 1, "theta": 110.10, "ktheta": 288.69600},
        "HAL1,CTL1,CTL3":   {"id":29, "func": 1, "theta": 110.10, "ktheta": 288.69600},
        "HAL2,CTL2,CTL1":   {"id":30, "func": 1, "theta": 110.10, "ktheta": 221.75200},
        "HAL2,CTL2,CH":     {"id":31, "func": 1, "theta": 110.10, "ktheta": 221.75200},
        "HAL2,CTL2,CTL2":   {"id":32, "func": 1, "theta": 110.10, "ktheta": 221.75200},
        "HAL2,CTL2,CTL3":   {"id":33, "func": 1, "theta": 110.10, "ktheta": 289.53280},
        "HAL3,CTL3,CTL1":   {"id":34, "func": 1, "theta": 110.10, "ktheta": 279.74224},
        "HAL3,CTL3,CTL2":   {"id":35, "func": 1, "theta": 110.10, "ktheta": 289.53280},
        "HAL3,CTL3,CTL3":   {"id":36, "func": 1, "theta": 110.10, "ktheta": 313.80000},
        "CTL1,CTL1,CTL1":   {"id":37, "func": 1, "theta": 111.00, "ktheta": 446.43280},
        "CTL1,CTL1,CTL2":   {"id":38, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL1,CTL1,CTL3":   {"id":39, "func": 1, "theta": 108.50, "ktheta": 446.43280},
        "CTL1,CTL2,CTL1":   {"id":40, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL1,CTL2,CTL2":   {"id":41, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL1,CTL2,CTL3":   {"id":42, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL2,CTL1,CTL2":   {"id":43, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL2,CTL1,CTL3":   {"id":44, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL2,CTL2,CTL2":   {"id":45, "func": 1, "theta": 113.60, "ktheta": 488.27280},
        "CTL2,CTL2,CTL3":   {"id":46, "func": 1, "theta": 115.00, "ktheta": 485.34400},
        "O2L,SL,O2L":       {"id":47, "func": 1, "theta": 109.47, "ktheta":1087.84000},
        "O2L,SL,OSL":       {"id":48, "func": 1, "theta":  98.00, "ktheta": 711.28000},
        "CTL2,OSL,SL":      {"id":49, "func": 1, "theta": 109.00, "ktheta": 125.52000},
        "CTL3,OSL,SL":      {"id":50, "func": 1, "theta": 109.00, "ktheta": 125.52000},
        "SL,CTL1,HAL1":     {"id":51, "func": 1, "theta": 110.10, "ktheta": 288.69600},
        "SL,CTL1,CL":       {"id":52, "func": 1, "theta": 108.00, "ktheta": 435.13600},
        "SL,CTL1,CTL2":     {"id":53, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL1,CL,OSL":      {"id":54, "func": 1, "theta": 109.00, "ktheta": 460.24000},
        "O2L,SL,CTL1":      {"id":55, "func": 1, "theta":  98.00, "ktheta": 711.28000},
        "HAL1,CTL1,CL":     {"id":56, "func": 1, "theta": 109.50, "ktheta": 276.14400},
        "CTL3,CT0,CTL3":    {"id":57, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "HAL2,CTL2,CT0":    {"id":58, "func": 1, "theta": 110.10, "ktheta": 221.75200},
        "CTL3,CT0,CTL2":    {"id":59, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "HAL3,CTL3,CT0":    {"id":60, "func": 1, "theta": 110.10, "ktheta": 279.74224},
        "CT0,CTL2,CTL1":    {"id":61, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "CTL3,CTL1,CTL3":   {"id":62, "func": 1, "theta": 113.50, "ktheta": 488.27280},
        "HT,OT,HT":         {"id":63, "func": 1, "theta": 104.52, "ktheta": 460.24000},
        "C3b,C,C3b":        {"id":64, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "C3b,C,CH2":        {"id":65, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "C3b,CH,CH2":       {"id":66, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "C3b,CH,C3b":       {"id":67, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "C,CH2,CH":         {"id":68, "func": 1, "theta": 114.00, "ktheta": 519.65690},
        "C3c,CH,CH2":       {"id":69, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "CH2,CH,CH2":       {"id":70, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "CTL2,CH,CH2":      {"id":71, "func": 1, "theta": 109.47, "ktheta": 519.65690},
        "C3a,CH2,CH":       {"id":72, "func": 1, "theta": 114.00, "ktheta": 519.65690},
        "CH,CH2,CH2":       {"id":73, "func": 1, "theta": 114.00, "ktheta": 519.65690},
        "CH2,CH2,CH2":      {"id":74, "func": 1, "theta": 114.00, "ktheta": 519.65690},
        "CH2,CH2,C3a":      {"id":75, "func": 1, "theta": 114.00, "ktheta": 519.65690}
        }

Dihedrals = {
        "SL,CTL1,CL,OBL"    :   {"id": 1, "func": 1, "theta": 180.000, "fc": 0.418, "mult": 6},
        "SL,CTL1,CL,OSL"    :   {"id": 2, "func": 1, "theta": 180.000, "fc": 1.255, "mult": 1},#weird
        "SL,CTL1,CTL2,CL"   :   {"id": 3, "func": 1, "theta": 180.000, "fc": 5.858, "mult": 1},
        "CTL1,CL,OSL,CTL2"  :   {"id": 4, "func": 1, "theta": 180.000, "fc":17.154, "mult": 2},
        "CTL1,CTL2,CL,OBL"  :   {"id": 5, "func": 1, "theta": 000.000, "fc": 0.000, "mult": 0},
        "CTL1,CTL2,CL,OSL"  :   {"id": 6, "func": 1, "theta": 180.000, "fc": 1.255, "mult": 1},#weird
        "CTL2,CTL1,CL,OBL"  :   {"id": 7, "func": 1, "theta": 000.000, "fc": 0.418, "mult": 0},
        "CTL2,CTL1,CL,OSL"  :   {"id": 8, "func": 1, "theta": 180.000, "fc": 1.255, "mult": 1},
        "CTL2,CL,OSL,CTL2"  :   {"id": 9, "func": 1, "theta": 000.000, "fc": 0.000, "mult": 0},
        "CL,OSL,CTL2,CH"    :   {"id":10, "func": 1, "theta": 180.000, "fc": 5.858, "mult": 1},
        "CL,CTL1,CTL2,CL"   :   {"id":11, "func": 1, "theta": 000.000, "fc": 0.000, "mult": 0},
        "OBL,CL,OSL,CTL2"   :   {"id":12, "func": 1, "theta": 180.000, "fc": 8.075, "mult": 1},
        "OSL,CTL2,CH,CH2"   :   {"id":13, "func": 1, "theta": 000.000, "fc": 0.000, "mult": 0},
        "CTL2,CH,CH2,CH2"   :   {"id":14, "func": 3, "C0": 12.4977, "C1":-14.5512, "C2":-1.15671, "C3": 14.9861, "C4": 0.0000, "C5": 0.0000},
        "CTL2,CH,CH2,C3a"   :   {"id":15, "func": 3, "C0": 12.4977, "C1":-14.5512, "C2":-1.15671, "C3": 14.9861, "C4": 0.0000, "C5": 0.0000},
        "CH,CH2,CH2,CH2"    :   {"id":16, "func": 3, "C0": 5.67474, "C1": 6.91717, "C2": 0.56697, "C3":-13.1589, "C4": 0.0000, "C5": 0.0000},
        "CH2,CH2,CH2,C3a"   :   {"id":17, "func": 3, "C0": 5.67474, "C1": 6.91717, "C2": 0.56697, "C3":-13.1589, "C4": 0.0000, "C5": 0.0000},
        "CH2,CH,CH2,C3a"    :   {"id":18, "func": 3, "C0": 12.4977, "C1":-14.5512, "C2":-1.15671, "C3": 14.9861, "C4": 0.0000, "C5": 0.0000},
        "CH2,CH,CH2,CH2"    :   {"id":19, "func": 3, "C0": 12.4977, "C1":-14.5512, "C2":-1.15671, "C3": 14.9861, "C4": 0.0000, "C5": 0.0000},
        "C,CH2,CH,C3b"      :   {"id":20, "func": 3, "C0": 6.80002, "C1": 20.4001, "C2": 0.00000, "C3":-27.2001, "C4": 0.0000, "C5": 0.0000},
        "CH,CH2,C,C3b"      :   {"id":21, "func": 3, "C0": 12.4977, "C1":-14.5512, "C2":-1.15671, "C3": 14.9861, "C4": 0.0000, "C5": 0.0000},
        }

snapshot = hoomd.data.make_snapshot(N=12727, 
        box=hoomd.data.boxdim(Lx=75,Ly=75,Lz=75),
        particle_types=["CTL1", "CTL2", "CTL3",
            "SL", "O2L", "HAL1", "HAL2", "HAL3",
            "CL", "OBL", "OSL", "CT0", "SOD", "HT",
            "OT", "C3a", "C3b", "C3c", "CH2", "CH",
            "C", "ZRS", "CLS"],
        bond_types=["CTL3,CL",  "CTL2,CL",  "CTL1,CL",  "OBL,CL",   "OSL,CL",
            "CTL1,HAL1", "CTL2,HAL2", "CTL3,HAL3", "CTL3,OSL", "CTL2,OSL",
            "CTL1,OSL", "CTL1,CTL1", "CTL1,CTL2", "CH,CTL2",  "CTL1,CTL3",
            "CTL2,CTL2", "CTL2,CTL3", "CTL3,CTL3", "SL,O2L",   "SL,OSL",
            "SL,CTL1",  "CT0,CTL3", "CT0,CTL2", "HT,HT", "HT,OT",    "CH,CH2",
            "CH2,C",    "C3b,C",    "C3b,CH",   "CH,C3c", "CH2,CH2",
            "CH2,C3a"],
        angle_types=["OBL,CL,CTL3",      "OBL,CL,CTL2",      "OBL,CL,CTL1",
            "OSL,CL,OBL", "CL,OSL,CTL1",      "CL,OSL,CTL2", "CL,OSL,CTL3",
            "HAL2,CTL2,CL", "HAL3,CTL3,CL", "CTL2,CTL2,CL",     "CTL3,CTL2,CL",
            "CTL2,CTL1,CL", "CTL1,CTL2,CL",     "OSL,CL,CTL3", "OSL,CL,CTL2",
            "OSL,CTL1,CTL2", "OSL,CTL1,CTL3", "OSL,CTL2,CTL1", "OSL,CTL2,CH",
            "OSL,CTL2,CTL2", "OSL,CTL2,CTL3", "HAL2,CTL2,HAL2",
            "HAL3,CTL3,HAL3", "HAL1,CTL1,OSL", "HAL2,CTL2,OSL",
            "HAL3,CTL3,OSL", "HAL1,CTL1,CTL1", "HAL1,CTL1,CTL2",
            "HAL1,CTL1,CTL3", "HAL2,CTL2,CTL1", "HAL2,CTL2,CH",
            "HAL2,CTL2,CTL2", "HAL2,CTL2,CTL3", "HAL3,CTL3,CTL1",
            "HAL3,CTL3,CTL2", "HAL3,CTL3,CTL3", "CTL1,CTL1,CTL1",
            "CTL1,CTL1,CTL2", "CTL1,CTL1,CTL3", "CTL1,CTL2,CTL1",
            "CTL1,CTL2,CTL2", "CTL1,CTL2,CTL3", "CTL2,CTL1,CTL2",
            "CTL2,CTL1,CTL3", "CTL2,CTL2,CTL2", "CTL2,CTL2,CTL3", "O2L,SL,O2L",
            "O2L,SL,OSL", "CTL2,OSL,SL",      "CTL3,OSL,SL", "SL,CTL1,HAL1",
            "SL,CTL1,CL", "SL,CTL1,CTL2",     "CTL1,CL,OSL", "O2L,SL,CTL1",
            "HAL1,CTL1,CL", "CTL3,CT0,CTL3", "HAL2,CTL2,CT0", "CTL3,CT0,CTL2",
            "HAL3,CTL3,CT0", "CT0,CTL2,CTL1", "CTL3,CTL1,CTL3",   "HT,OT,HT",
            "C3b,C,C3b", "C3b,C,CH2", "C3b,CH,CH2",       "C3b,CH,C3b",
            "C,CH2,CH", "C3c,CH,CH2", "CH2,CH,CH2",       "CTL2,CH,CH2",
            "C3a,CH2,CH", "CH,CH2,CH2", "CH2,CH2,CH2", "CH2,CH2,C3a"],
        dihedral_types=["SL,CTL1,CL,OBL", "SL,CTL1,CL,OSL", "SL,CTL1,CTL2,CL",
            "CTL1,CL,OSL,CTL2", "CTL1,CTL2,CL,OBL", "CTL1,CTL2,CL,OSL",
            "CTL2,CTL1,CL,OBL", "CTL2,CTL1,CL,OSL", "CTL2,CL,OSL,CTL2",
            "CL,OSL,CTL2,CH", "CL,CTL1,CTL2,CL", "OBL,CL,OSL,CTL2",
            "OSL,CTL2,CH,CH2", "CTL2,CH,CH2,CH2", "CTL2,CH,CH2,C3a",
            "CH,CH2,CH2,CH2", "CH2,CH2,CH2,C3a", "CH2,CH,CH2,C3a",
            "CH2,CH,CH2,CH2", "C,CH2,CH,C3b", "CH,CH2,C,C3b"])

dirwithstuff = os.path.abspath(sys.argv[1])
justText = os.path.join(dirwithstuff,'justname.txt')
justCoords = os.path.join(dirwithstuff,'justcoords.txt') 
with open(justText,'r') as nameFile:
    snapshotNames = [line.strip() for line in nameFile.readlines()]
snapshot.particles.position[:] = np.genfromtxt(justCoords,delimiter=',')
snapshot.particles.typeid[:] = [Particles[atomToParticle[atom]["type"]]["id"] for atom in snapshotNames]
snapshot.particles.mass[:] = [Particles[atomToParticle[atom]["type"]]["mass"] for atom in snapshotNames]
snapshot.particles.charge[:] = [Particles[atomToParticle[atom]["type"]]["charge"] for atom in snapshotNames]

totalBonds,totalAngles,totalDihedrals = [],[],[]
totalBondIds,totalAngleIds,totalDihedralIds = [],[],[]
atomIndex = 0
for atom in snapshotNames:
    for bond in atomToParticle[atom]["bonds"]:
        totalBonds.append(bond+atomIndex)
    for angle in atomToParticle[atom]["angles"]:
        totalAngles.append(angle+atomIndex)
    for dih in atomToParticle[atom]["dihedrals"]:
        totalDihedrals.append(dih+atomIndex)
    atomIndex += 1
totalBonds = np.array(totalBonds)
totalAngles = np.array(totalAngles)
totalDihedrals = np.array(totalDihedrals)

for bond in totalBonds:
    try:
        totalBondIds.append(Bonds[','.join([atomToParticle[snapshotNames[bond[0]]]["type"],atomToParticle[snapshotNames[bond[1]]]["type"]])]["id"])
    except KeyError:
        totalBondIds.append(Bonds[','.join([atomToParticle[snapshotNames[bond[1]]]["type"],atomToParticle[snapshotNames[bond[0]]]["type"]])]["id"])
totalBondIds = np.array(totalBondIds)

for angle in totalAngles:
    try:
        totalAngleIds.append(Angles[','.join([atomToParticle[snapshotNames[angle[0]]]["type"],atomToParticle[snapshotNames[angle[1]]]["type"],atomToParticle[snapshotNames[angle[2]]]["type"]])]["id"])
    except KeyError:
        totalAngleIds.append(Angles[','.join([atomToParticle[snapshotNames[angle[2]]]["type"],atomToParticle[snapshotNames[angle[1]]]["type"],atomToParticle[snapshotNames[angle[0]]]["type"]])]["id"])
totalAngleIds = np.array(totalAngleIds)

#To get names of dihedral types
#dihnames = OrderedSet([])
#for dih in totalDihedrals:
#    dihname = ','.join([atomToParticle[snapshotNames[dih[0]]]["type"],atomToParticle[snapshotNames[dih[1]]]["type"],atomToParticle[snapshotNames[dih[2]]]["type"],atomToParticle[snapshotNames[dih[3]]]["type"]])
#    dihnames.add(dihname)
#print('\n'.join(['"%s",'%(name) for name in dihnames]))
for dih in totalDihedrals:
    totalDihedralIds.append(Dihedrals[','.join([atomToParticle[snapshotNames[dih[0]]]["type"],atomToParticle[snapshotNames[dih[1]]]["type"],atomToParticle[snapshotNames[dih[2]]]["type"],atomToParticle[snapshotNames[dih[3]]]["type"]])]["id"])
totalDihedralIds = np.array(totalDihedralIds)

snapshot.bonds.resize(totalBonds.shape[0])
snapshot.bonds.group[:] = totalBonds
snapshot.bonds.typeid[:] = totalBondIds - 1 # Dictionary indexing started at 1

snapshot.angles.resize(totalAngles.shape[0])
snapshot.angles.group[:] = totalAngles
snapshot.angles.typeid[:] = totalAngleIds - 1 # Dictionary indexing started at 1

snapshot.dihedrals.resize(totalDihedrals.shape[0])
snapshot.dihedrals.group[:] = totalDihedrals
snapshot.dihedrals.typeid[:] = totalDihedralIds - 1 # Dictionary indexing started at 1

################################################################
# Creating Initial Stucture
################################################################

hoomd.init.read_snapshot(snapshot)
hoomd.dump.gsd("initialization.gsd")

