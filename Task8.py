#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16,8]


X = np.array([[1,1.1,0],[1,1.1,0],
              [2,0.8,1],[2,0.8,1],
              [1,0.9,0.1],[1,0.9,0.1]])

Y = np.array([[0.86, 0.81],[0.86,0.81],
              [1.78,1.04],[1.78,1.04],
              [0.83,0.71],[0.83,0.71]])
nPoints = X.shape[0]
Xavg = np.mean(X, axis=0)          #Compute mean
B = X - np.tile(Xavg, (nPoints,1))   #Mean-Subtracted data
B = X
U, S, VT = np.linalg.svd(B)
S = np.diag(S)
#%%
atrb = 2
U1  = U[:,:atrb]
VT1 = VT[:,:atrb]
S1 = S[:atrb,:atrb]
#a = Y   @ U1.T  @ np.linalg.inv(S1) @ VT1
a = VT1 @ np.linalg.inv(S1) @ U1.T @ Y

#%%
ax2 = fig.add_subplot(122)
ax2.plot(X[0,:],X[1,:], '.', color='k')   # Plot data to overlay PCA
ax2.grid()
plt.xlim((-6, 8))
plt.ylim((-6,8))

theta = 2 * np.pi * np.arange(0,1,0.01)
# 1-std confidence interval
Xstd = U @ np.diag(S) @ np.array([np.cos(theta),np.sin(theta)])

ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1] + Xstd[1,:],'-',color='r',linewidth=3)
ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1] + 2*Xstd[1,:],'-',color='r',linewidth=3)
ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1] + 3*Xstd[1,:],'-',color='r',linewidth=3)

# Plot principal components U[:,0]S[0] and U[:,1]S[1]
ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,0]*S[0]]),
         np.array([Xavg[1], Xavg[1]+U[1,0]*S[0]]),'-',color='cyan',linewidth=5)
ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,1]*S[1]]),
         np.array([Xavg[1], Xavg[1]+U[1,1]*S[1]]),'-',color='cyan',linewidth=5)

plt.show()

#%% PCA and PCR

def mypca(Y, X, a):
    num_col, num_rows = X.shape
    nPoints = num_rows
    Xavg = np.mean(X, axis=1)
    B = X - np.tile(Xavg, (nPoints,1)).T
    U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)
    
  
    





