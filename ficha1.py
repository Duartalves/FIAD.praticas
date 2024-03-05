import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import detrend

#--- Ler ficheiro com a série temporal
#--- Contém: tempas temperaturas médias
# série temporal
x1 = np.genfromtxt("lisbon_temp_fmt.txt", dtype='float')

N=len(x1) # Comprimento da série temporal
print('N = ',N)
n = np.arange(N) # escala temporal em meses

#--- Representação gráfica
plt.figure(figsize=(12,5))
plt.plot(n,x1,'-+',label='Série temporal')
plt.xlabel('n [meses]')
plt.ylabel('T [ºC]')
plt.title('Temperatura em Lisboa desde janeiro de 1980')
plt.legend(loc='upper left')
plt.show()

# Ex 1.2
#---Verifica a existência de NaN
haNaN=np.isnan(x1).any() # Há NaN?
print("Há NaN? -", haNaN)

indNaN=np.where(np.isnan(x1))[0] # Índice dos elementos com NaN
print("indNaN = ",indNaN)
print("x1[indNaN] = ",x1[indNaN])
#---Reconstruir as lacunas usando extrapolação
x1r=np.array(x1, copy=True)

if haNaN:
    ind = indNaN
    for k in range(len(ind)):
        nn=n[ind[k]-4:ind[k]] # admitindo que não há NaNs no início
        xx=x1r[ind[k]-4:ind[k]]
        f=interp1d(nn,xx,fill_value='extrapolate')
        x1r[ind[k]]=f(n[ind[k]])
print("x1r[indNaN] = ",x1r[indNaN])

#--- Representação gráfica
plt.figure(figsize=(12,5))
plt.plot(n,x1,'-+',label='Série temporal')
plt.plot(n,x1r,'-o',label='Série temporal sem NaN')
plt.legend(loc='upper left')
plt.ylabel('T [ºC]')
plt.xlabel('n [meses]')
plt.show()

# Ex 1.3
#---Média, desvio padrão e correlação
# - Funções: mean, std e corrcoef
mu1 = x1r.mean()
print("mu1 = ", mu1)
sigma1 = x1r.std()
print("sigma1 = ", sigma1)

# Determinar temperaturas nas décadas de 80 e 90
ix80=0
ix89=10*12-1
ix90=ix89+1
ix99=ix90+10*12-1

temp80=x1r[ix80:ix89+1]
temp90=x1r[ix90:ix99+1]

print('Correlação entre as duas séries temporais sem NaN:')
corr=np.corrcoef(temp80,temp90)
print("corrcoef = ", corr)

# Ex 1.4
#---Verifica outliers
indoutl1=np.where(abs(x1r - mu1) > 3*sigma1)[0] # Índice dos outliers
nout1 = len(indoutl1) # número de outliers
print("Quantos outliers? -", nout1)
print("indoutl1 = ",indoutl1)
print("x1r[indoutl1] = ",x1[indoutl1])

x1ro=np.array(x1r,copy=True) # Substituição dos outliers
if nout1:
    for k in range(len(indoutl1)):
        if x1ro[indoutl1[k]] > mu1:
            x1ro[indoutl1[k]] = mu1 + 2.5*sigma1
        else:
            x1ro[indoutl1[k]] = mu1 - 2.5*sigma1

#--- Representação gráfica dos resultados
plt.figure(figsize=(12,5))
plt.plot(n,x1,'-+',label='Série temporal')
plt.plot(n,x1r,'-o',label='Série temporal sem NaN')
plt.plot(n,x1ro,'-d',label='Série temporal sem outliers')
plt.legend(loc='upper left')
plt.ylabel('T [ºC]')
plt.xlabel('n [meses]')
plt.show()

