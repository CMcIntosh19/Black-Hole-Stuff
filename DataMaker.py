# -*- coding: utf-8 -*-
"""
Created on Tue May 24 00:26:28 2022

@author: hepiz
"""

import numpy as np
import MainLoops as ml

orbs = [[[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]],
        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]],
        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]],
        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]],
        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]]]
        

circ_v = 0.00035355339059327376 #borderline, plunged quickly at high mu
circ_v = 0.000353554

initial  = np.array([ [0.00, 200.0, 1.57079633, 0.00, 1.00188029, 0.00, 0.00, circ_v*0.2805688502888831] ])

for i in range(3):
    for j in range(3):
        for k in range(3):
            spin = 0.0 if (i == 0) else 10**(i-3)
            mu = 0.0 if (j == 0) else 10**(j-3)
            xi = 90.0 - k*15
            label = str(spin) + "/" + str(mu) + "/" + str(xi)
            print("calculating " + label)
            orbs[i][j][k] = ml.inspiral_long(initial, 1, spin, mu, 1, 20000, 0.1, True, 10**(-11), 90, xi, label)



""