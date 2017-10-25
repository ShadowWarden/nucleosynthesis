# nucleosynthesis/nucleosynthesis.py
#
# Omkar H. Ramachandran
# omkar.ramachandran@colorado.edu
#
# A simple population dynamics simulation of the elements involved in
# nucleosynthesis. All decay chains and half lives were pulled from
# https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html
#

import numpy as np
import math

# Change STEP to desired temporal step size
STEP = 119e-3
SECOND = 1./STEP
MILLISECOND = SECOND*1e-3
MINUTE = 60*SECOND
DAY = 86400*SECOND
YEAR = 3.154e7*SECOND

# Rules : 0 = stable, 1 = beta+, 2 = beta-, 3 = proton,
# 4 = neutron, 5 = alpha
Rules = np.array([
        [0,0,2,4,4,4,4],
        [1,0,0,4,2,4,2],
        [3,3,3,0,0,2,2],
        [3,3,3,1,5,0,2],
        [3,3,3,1,3,0,0],
        [3,3,3,1,1,1,0]
        ])

# Matrix of atomic radii
Radius = np.array([
        np.ones(7)*53,
        np.ones(7)*31,
        np.ones(7)*167,
        np.ones(7)*112,
        np.ones(7)*87,
        np.ones(7)*67
        ])
Radius *= 1e-12

# Matrix of half lives: math.inf reprasents stable isotopes
Thalf = np.array([
        [math.inf,math.inf,12.3*YEAR,0,0,0,0],
        [0,math.inf,math.inf,0,806.7*MILLISECOND,0,119*MILLISECOND],
        [0,0,0,math.inf,math.inf,839*MILLISECOND,178.3*MILLISECOND],
        [0,0,0,532*DAY,0,math.inf,1.5e6*YEAR],
        [0,0,0,0,770*MILLISECOND,0,math.inf],
        [0,0,0,126.5*MILLISECOND,19.2*SECOND,20*MINUTE,math.inf]
        ])
# Performance hack. Reduce number of divisions
Thalf = 1/Thalf

# Number of timesteps
Nt = 1000
# Radius of 20 Carbon Atoms
Area = 20*np.pi*Radius[5][0]**2
# Performance hack. Change division to multiplication
onedArea = 1./Area
# Starting number of Hydrogen Atoms
NHydrogen = 1e6

def decay_product(Z,N):
    """ Look up decay rule and return the decay product """
    rule = Rules[Z,N]
#    print(rule)
    if(rule == 0):
        # Stable. Return Z,N. No decay
        return np.array([Z,N])
    elif(rule == 1):
        # Beta plus. Return Z-1, N+1
        return np.array([Z-1,N+1])
    elif(rule == 2):
        # Beta minus. Return Z+1,N-1
        return np.array([Z+1,N-1]) 
    elif(rule == 3):
        # Proton emission. Return Z-1,N
        return np.array([Z-1,N])
    elif(rule == 4):
        # Neutron emission. Return Z, N-1
        return np.array([Z,N-1])
    elif(rule == 5):
        # Alpha Decay. Return Z-2,N-2
        return np.array([Z-2,N-2])

def fusion_product(atom1,atom2):
    """ Return Fusion product 
        Input:
        atom1 -> np.array([Z1,N1])
        atom2 -> np.array([Z2,N2])
    """
    if(atom1[0]+atom2[0]+1 > 5):
        # Goes beyond Carbon. Not allowed
        return atom1
    if(atom1[1]+atom2[1] > 6):
        # Goes beyond 6 neutrons. Not allowed
        return atom1
    return np.array([atom1[0]+atom2[0]+1,atom1[1]+atom2[1]])

# Define population matrix
pop = np.zeros([Nt+1,6,7])
pop[0,0,0] = NHydrogen
pop[0,0,1] = 1000

for t in range(Nt):
    # Find points where pop != 0. Greatly reduces computation time
    ii = np.where(pop[t] != 0)
    pop[t+1] = pop[t,:,:]
    for i in range(np.shape(ii)[1]):
        atom1 = np.array([ii[0][i],ii[1][i]])
        for j in range(len(ii)):
            atom2 = np.array([ii[0][j],ii[1][j]])
            # Compute Fusion chain first
            prod = fusion_product(atom1,atom2)
    #        print(prod,atom1,atom2,t)
            if((prod == atom1).all()):
                # No fusion possible
                continue
            # Compute new population. No dependence on Thalf
            diff = np.min([pop[t,atom1[0],atom1[1]],pop[t,atom2[0],atom2[1]]])*np.pi*(Radius[atom1[0],atom1[1]]+Radius[atom2[0],atom2[1]])**2*onedArea
            #print(diff,t)
            pop[t+1,prod[0],prod[1]] += diff/2.
      #      print(prod,pop[t+1,prod[0],prod[1]])
            pop[t+1,atom1[0],atom1[1]] -= diff/2.
            pop[t+1,atom2[0],atom2[1]] -= diff/2.
        # Compute decay. Need to do this for each atom
        prod = decay_product(atom1[0],atom1[1])
        # Probability's one, Thalf determines rate. If negative, reset to 0 and remove the
        # difference from pop[t+1]
     #   print(Thalf[atom1[0],atom1[1]]*np.log(2),atom1,t)
        if(Thalf[atom1[0],atom1[1]]*np.log(2) > 1.):
        # pop[t] goes to zero. Just add pop[t] to pop[t+1] and reset pop[t] to zero
     #       print(atom1)
            pop[t+1,prod[0],prod[1]] += pop[t,atom1[0],atom1[1]]
            pop[t+1,atom1[0],atom1[1]] = 0
        else:
        # pop[t] does not go to zero. Use the standard formula
            diff = pop[t,atom1[0],atom1[1]]*Thalf[atom1[0],atom1[1]]*np.log(2)
            pop[t+1,prod[0],prod[1]] += diff
            pop[t+1,atom1[0],atom2[0]] -= diff
