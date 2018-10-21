import nashpy as nash
import numpy as np

# Matching Pennies
A = np.array([[1,-1], [-1,1]])
matching_pennies = nash.Game(A)
matching_pennies

# Prisoners Dilemma
A = np.array([[3,0], [5,1]])
B = np.array([[3,5], [0,1]])
prisoners_dilemma = nash.Game(A, B)
prisoners_dilemma