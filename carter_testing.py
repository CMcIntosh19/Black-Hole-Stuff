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

t = 100000
a = 0.0
initial  = np.array([ [0.00, 29.0, np.pi/2, 0.00, 0.98, 5.1, 0.0] ])
test01notc2 = inspiral_long(initial, 1, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, 'C0 sago near_circle', verbose=True)
test02notc2 = inspiral_long1(initial, 1, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, 'C0 peters eq near_circle', verbose=True)
test03notc2 = inspiral_long2(initial, 1, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, 'C0 peters int Lz near_circle', verbose=True)
test04notc2 = inspiral_long3(initial, 1, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, 'C0 peters int Lall near_circle', verbose=True)
test05notc2 = inspiral_long4(initial, 1, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, 'C0 peters bigA Lall near_circle', verbose=True)
physplots([test01notc2, test02notc2, test03notc2, test04notc2, test05notc2], merge=False)
orthoplots([test01notc2, test02notc2, test03notc2, test04notc2, test05notc2], merge=False)