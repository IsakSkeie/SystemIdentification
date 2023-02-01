#%%
import numpy as np
from scipy import linalg
 #%% A)
 y0 = 1
 y1 = 0.9
 y2 = 0.81
 y3 = 0.729
 y4 = 0.6561
 
 Y1 = np.array([[y0,y1,y2], [y1,y2,y3]])
 Y2 = np.array([[y1,y2, y3], [y2,y3, y4]])
 
 u, s, v    = linalg.svd(Y2, full_matrices= True)
# %%
