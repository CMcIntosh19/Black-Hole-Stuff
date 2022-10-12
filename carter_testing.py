# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:34:51 2022

@author: hepiz
"""
import numpy as np
import MetricMath as mm
import MainLoops as ml
import matplotlib.pyplot as plt
import OrbitPlotter as op

'''
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 4.0, 0.0] ])
test0_1 = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 sago', verbose=True)
test0_4 = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 peters int Lall', verbose=True)
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 3.84, 0.0] ])
test0_1d = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 sago deep', verbose=True)
test0_4d = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 peters int Lall deep', verbose=True)
'''
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 5.24, 0.0] ])
test0_1nc2 = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C0 sago near_circle', verbose=True)
test0_4nc2 = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C0 peters int Lall near_circle', verbose=True)
'''
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 4.0, 0.01] ])
test_01_1 = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 sago', verbose=True)
test_01_4 = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 peters int Lall', verbose=True)

initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 3.84, 0.01] ])
test_01_1d = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 sago deep', verbose=True)
test_01_4d = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 peters int Lall deep', verbose=True)
'''
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 5.24, 0.01] ])
test_01_1nc2 = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 sago near_circle', verbose=True)
test_01_4nc2 = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_01 peters int Lall near_circle', verbose=True)
'''
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 4.0, 0.1] ])
test_1_1 = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_1 sago', verbose=True)
test_1_4 = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_1 peters int Lall', verbose=True)
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 3.84, 0.1] ])
test_1_1d = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_1 sago deep', verbose=True)
test_1_4d = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_1 peters int Lall deep', verbose=True)
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 5.24, 0.1] ])
test_1_1nc = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_1 sago near_circle', verbose=True)
test_1_4nc = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C_1 peters int Lall near_circle', verbose=True)
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 4.0, 1.0] ])
test1_1 = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C1 sago', verbose=True)
test1_4 = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C1 peters int Lall', verbose=True)
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 3.84, 1.0] ])
test1_1d = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C1 sago deep', verbose=True)
test1_4d = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C1 peters int Lall deep', verbose=True)
initial  = np.array([ [0.00, 23.5, np.pi/2, 0.00, 0.98, 5.24, 1.0] ])
test1_1nc = ml.inspiral_long(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C1 sago near_circle', verbose=True)
test1_4nc = ml.inspiral_long3(initial, 1, 0.0, 10**(-4), 1, 5000, 0.1, True, 10**(-13), 90, 90, 'C1 peters int Lall near_circle', verbose=True)
'''