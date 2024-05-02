# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from numpy import mean, array, where
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import seaborn as sns
from numpy import linalg as la
import math as mat
import scipy as sc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


                            #############################################
                            #             Ejercicio 04                  #
                            #############################################

dat = {'Mass':[5.526,10.401,9.213,8.953,7.063,6.610,11.273,2.447,15.493,9.004,8.199,6.601,7.622,10.067,10.091,10.888,7.610,7.733,12.015,10.049,5.149,9.158,12.132,6.978,6.890],
       'SVL':[59,75,69,67.5,62,62,74,47,86.5,69,70.5,64.5,67.5,73,73,77,61.5,66.5,79.5,74,59.5,68,75,66.5,63],
     'HLS':[113.5,142,124,125,129.5,123,140,97,162,126.5,136,116,135,136.5,135.5,139,118,133.5,150,137,116,123,141,117,117]}
datos = pd.DataFrame(dat)

#---------------------------#
#   datos estandarizados    #
#---------------------------#
datos_estan = StandardScaler().fit_transform(datos)
datos_estan = pd.DataFrame(datos_estan)
datos_estan2 = datos_estan
datos_estan = datos_estan.rename(columns={0: 'Mass', 1: 'SVL',2:'HLS'})  #datos estandarizados
datos_estan


datos_estan = StandardScaler().fit_transform(datos)
mat_covt=np.cov(np.transpose(datos_estan), bias=True)

sexo = [0,1,0,0,1,0,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0]
datos_estan2["Sexo"] = pd.Series(sexo, index=datos_estan2.index)

datos_hem = datos_estan[datos_estan2['Sexo'] == 0] # Dataframe de hembras
datos_mac= datos_estan[datos_estan2['Sexo'] == 1]  # DataFrame de machos

#---------------------------#
#   Matriz de Covarianzas   #
#---------------------------#
hembras_cov=np.cov(np.transpose(datos_hem), bias=True)  # datos_hem.cov()
machos_cov=np.cov(np.transpose(datos_mac), bias=True)   # datos_mac.cov()

print("Matriz de covarianzas de poblacion entera","\n",mat_covt, "\n"*3,
"Matriz de covarianzas de hembras \n", hembras_cov, "\n"*3,
"Matriz de covarianzas de machos \n",machos_cov)


#------------------------------#
#   Descomposicion Espectral   #
#------------------------------#
Gamma1, Lambda1 = la.eig(mat_covt)      # Matriz Total
Gamma2, Lambda2 = la.eig(hembras_cov)   # Matriz de Hembras
Gamma3, Lambda3 = la.eig(machos_cov)    # Matriz de Machos


#-------------#
#   Grafica   #
#-------------#
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(datos_hem[:,0],datos_hem[:,1],datos_hem[:,2],c='r',marker='o', label="hembras")
ax.scatter(datos_mac[:,0],datos_mac[:,1],datos_mac[:,2], c='b',marker='^', label="machos")
plt.legend()
ax.set_xlabel("Mass")
ax.set_ylabel("SVL")
ax.set_zlabel("HLS")
plt.show()



import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from scipy.stats import chi2



fig = plt.figure(figsize=(8,8))    # Creamos la figura
ax = fig.add_subplot(111, projection='3d') 

#=================================#
#     Elipses de Confianza        #
#=================================#

## Definimos la primer Elipse
A = hembras_cov   # matriz de covarianzas para hembras 
A2 = machos_cov  

# ajustamos el centro de la elipse con  las medias de los datos
center = np.array(datos_hem.mean())  
center2 = np.array(datos_mac.mean()) 

# encontramos el radio y la rotacion 
U, s, rotation = linalg.svd(A)
U2, s2, rotation2 = linalg.svd(A2)
radii = np.sqrt( s*chi2.ppf(q=(1-0.05),df=3)/3)    ## tamaño de los semiejes
radii2 = np.sqrt( s2*chi2.ppf(q=(1-0.05),df=3)/3)  # chi-cuadrada, df=2, alfa=0.05


#------------------------#
#  cambio de cordenadas  #
#------------------------#
u = np.linspace(0.0, 2.0 * np.pi, 60)    
v = np.linspace(0.0, np.pi, 60)

x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

x2 = radii2[0] * np.outer(np.cos(u), np.sin(v))
y2 = radii2[1] * np.outer(np.sin(u), np.sin(v))
z2 = radii2[2] * np.outer(np.ones_like(u), np.cos(v))


# Los puntos de la Elipse
for i in range(len(x)):
    for j in range(len(x)):
      
      [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center  
      [x2[i,j],y2[i,j],z2[i,j]] = np.dot([x2[i,j],y2[i,j],z2[i,j]], rotation2) + center2

ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color='red', linewidth=0.1, alpha=0.3, shade=True)
ax.plot_surface(x2, y2, z2,  rstride=3, cstride=3,  color='b', linewidth=0.1, alpha=0.3, shade=True)
ax.scatter(datos_hem[:,0],datos_hem[:,1],datos_hem[:,2],c='r',marker='o', label="hembras")
ax.scatter(datos_hem[:,0],datos_hem[:,1],datos_hem[:,2], c='b',marker='^', label="machos")
plt.legend()
ax.set_xlabel("Mass")
ax.set_ylabel("SVL")
ax.set_zlabel("HLS")
plt.title("Elipses de confianza al 95%")

plt.show()













                            #############################################
                            #             Ejercicio 05                  #
                            #############################################
from numpy import array, where, mean, var, exp, sqrt, std, cov, pi, cos, sin

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.covariance import EllipticEnvelope

Xt = pd.read_csv('/home/guy/Downloads/time_series_1.csv')
Yt = pd.read_csv('/home/guy/Downloads/time_series_2.csv')

#---------------------------------------#
#      Graficas y Autocorrelaciones     #
#---------------------------------------#
plt.plot(Xt)
plt.plot(Yt)
plt.legend(['Serie 01','Serie 02'], loc='center left', bbox_to_anchor=(1, 0.5))

# Autocorrelaciones
plot_acf(Xdiff)
plot_pacf(Xdiff)
plot_acf(Xt)
plot_pacf(Xt)
plot_acf(Yt)
plot_pacf(Yt)


plt.plot(Xt.diff())

#---------------------------------------#
#            Ajuste de Modelo           #
#---------------------------------------#
Xdiff = Xt.diff()[1:]
Xdiff = Xdiff.diff()[1:]            # Segunda Diferencia
modXt = ARIMA(Xdiff,order=(3,0,0))
modFitXt = modXt.fit()
modFitXt.summary()

plt.plot(Xdiff)
plt.plot(modFitXt.fittedvalues)
# modFitXt.plot_predict()




#---------------------------------------#
#             Matriz de Datos           #
#---------------------------------------#
Mo = np.zeros((len(Xdiff), 3))
Mo = pd.DataFrame(Mo,columns=['A','B','C'])

# Llenamos
for k in range(0,len(Xdiff)):
    Mo.iloc[k,0] = Xdiff.iloc[k,0]
    if k<(len(Xdiff)-1):
        Mo.iloc[k,1] = Xdiff.iloc[k+1,0]
    if k<(len(Xdiff)-2):
        Mo.iloc[k,2] = Xdiff.iloc[k+2,0]



#---------------------------#
#   Datos Estandarizados    #
#---------------------------#
for k in ['A','B','C']:
    mT = mean( Mo[k] )    # Media
    dE = std( Mo[k] )     # Desviacion Estandar
    Mo[k] = (Mo[k]-mT)/dE


#------------#
# Covarianza #    
#------------#
from numpy import linalg
from mpl_toolkits import mplot3d

covMo = cov(Mo.T)
eigVal = linalg.eig(covMo)[0]
eigVec = linalg.eig(covMo)[1]

centroM = array(Mo.mean())
P,D,P= linalg.svd(covMo)
radioM = sqrt(D*chi2.ppf(0.95,df=3))

u = np.linspace(0,2*pi,100)    
v = np.linspace(0,pi,100)

x = radioM[0] * np.outer(np.cos(u), np.sin(v))
y = radioM[1] * np.outer(np.sin(u), np.sin(v))
z = radioM[2] * np.outer(np.ones_like(u), np.cos(v))

for i in range(len(x)):
    for j in range(len(x)):      
      [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], P) + centroM 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(Mo['A'],Mo['B'],Mo['C'],alpha=0.5)
ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color='red', linewidth=0.1, alpha=0.5, shade=True)

#ax.scatter(centroM[0],centroM[1],centroM[2],color='red')
ax.view_init(50, 60)



#=================================#
#    Metodo Envolvente Eliptica   #
#=================================#
    
from sklearn.covariance import EllipticEnvelope

outlier_detector = EllipticEnvelope(contamination=0.01)
outlier_detector.fit(Xdiff)
jj = outlier_detector.predict(Xdiff)


Ao = where(jj<1)[0]
plt.plot(np.arange(0,len(Xdiff),1),Xdiff)
plt.scatter(Ao, Xdiff.iloc[Ao], color='red')

plt.legend(['Serie Diferenciada','Datos Atipicos'], 
           loc='center left', bbox_to_anchor=(1, 0.5))






                            ###################################
                            #=================================#
                            #          Segunda Serie          #
                            #=================================#
# Yt = pd.read_csv('time_series_2.csv')

#---------------------------------------#
#      Graficas y Autocorrelaciones     #
#---------------------------------------#
plt.plot(Yt)
plt.legend(['Serie 02'], loc='center left', bbox_to_anchor=(1, 0.5))

#---------------------------------------#
#            Ajuste de Modelo           #
#---------------------------------------#
Ydiff = Yt.diff()[1:]
Ydiff = Ydiff.diff()[1:]            # Segunda Diferencia
modYt = ARIMA(Ydiff,order=(1,0,0))
modFitYt = modYt.fit()
modFitYt.summary()

plt.plot(Ydiff,color='steelblue')
plt.plot(modFitYt.fittedvalues,color='orange')
plt.legend(['Serie Diferenciada','Modelo Ajustado'], loc='center left', bbox_to_anchor=(1, 0.5))


#---------------------------------------#
#             Matriz de Datos           #
#---------------------------------------#
Moo = np.zeros((len(Ydiff), 2))
Moo = pd.DataFrame(Moo,columns=['A','B'])

# Llenamos
for k in range(0,len(Ydiff)):
    Moo.iloc[k,0] = Ydiff.iloc[k,0]
    if k<(len(Ydiff)-1):
        Moo.iloc[k,1] = Ydiff.iloc[k+1,0]



#---------------------------#
#   Datos Estandarizados    #
#---------------------------#
for k in ['A','B']:
    mT = mean( Moo[k] )    # Media
    dE = std( Moo[k] )     # Desviacion Estandar
    Moo[k] = (Moo[k]-mT)/dE
   

#------------#
# Covarianza #    
#------------#
from numpy import linalg
from mpl_toolkits import mplot3d
from scipy.stats import chi2

covMoo = cov(Moo.T)
eigVal2 = linalg.eig(covMoo)[0]
eigVec2 = linalg.eig(covMoo)[1]

centroM2 = array(Moo.mean())
P,D,P= linalg.svd(covMoo)    # Descomposicion en Valores Singulares
radioM2 = sqrt(D*chi2.ppf(0.95,df=3))


#=================================#
#      Grafica de la Elipsoide    #
#=================================#
u = np.linspace(0,2*pi,100)    
v = np.linspace(0,pi,100)

x = radioM2[0] * np.outer(np.cos(u), np.sin(v))
y = radioM2[1] * np.outer(np.sin(u), np.sin(v))

for i in range(len(x)):
    for j in range(len(x)):      
      [x[i,j],y[i,j]] = np.dot([x[i,j],y[i,j]], P) + centroM2 

plt.scatter(Moo['A'],Moo['B'],alpha=0.3)
#plt.plot([centroM2[0],centroM2[0]+1.61*eigVec2[0][0]], 
#          [centroM2[1],centroM2[1]+1.61*eigVec2[0][1]],
#          color='red')
#plt.plot([centroM2[0],centroM2[0]+1.61*eigVec2[1][0]], 
#          [centroM2[1],centroM2[1]+1.61*eigVec2[1][1]],
#          color='red')
plt.axis('equal')
plt.plot(x, y, color='red', alpha=0.1)



#=================================#
#    Otra Manera para rotar       #
#           la Elipse             #
#=================================#
L1 = eigVal2[0]
L2 = eigVal2[1]
xx = sqrt( L1 * chi2.ppf(0.95,df=3)) * cos(u)
yy = sqrt( L2 * chi2.ppf(0.95,df=3)) * sin(u)
plt.plot(xx,yy)

V = eigVec2[0]
W = eigVec2[1]
Angulo = np.arctan(W[1]/W[0])

# Matriz de Rotacion
Rot = array([[cos(Angulo), -sin(Angulo)],
             [sin(Angulo), cos(Angulo)]])
Pxy = array([xx,yy])

# Rotamos los puntos
xx,yy = np.matmul(Rot,Pxy)

#for k in range(0,len(xx)):
#    xx[k] = np.dot( [xx[k],yy[k]], Rot[0,]) 
#    yy[k] = np.dot( [xx[k],yy[k]], Rot[1,])
plt.plot(xx,yy)


#=================================#
#    Metodo Envolvente Eliptica   #
#=================================#
    
from sklearn.covariance import EllipticEnvelope

outlier_detector = EllipticEnvelope(contamination=0.01)
outlier_detector.fit(Ydiff)
jj = outlier_detector.predict(Ydiff)


Ao2 = where(jj<1)[0]
plt.plot(np.arange(0,len(Ydiff),1),Ydiff)
plt.scatter(Ao2, Ydiff.iloc[Ao2], color='red')

plt.legend(['Serie Diferenciada','Datos Atipicos'], 
           loc='center left', bbox_to_anchor=(1, 0.5))



