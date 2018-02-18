from __future__ import division, print_function
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.colors
import skfuzzy as fuzz
from mpl_toolkits.mplot3d import Axes3D

import math

def process_data(file,time_start,time_end,num_plot):
    data=np.loadtxt(file,usecols=range(13))
    time_start_find = np.argmax(data[:, 0] >= time_start)
    time_end_find = np.argmax(data[:, 0] >= time_end)
    if time_end_find == 0:
        time_end_find = -1
    time = data[time_start_find:time_end_find, 0]
    data = data[time_start_find:time_end_find,:]
    print('imported data ' + file, data.shape, ' time', time.shape, ' start : ', time[0], ' end : ', time[-1])
    axis[num_plot].plot(time, data[:,3])
    axis[num_plot+1].plot(time, data[:, 6])
    axis[num_plot+2].plot(time, data[:, 9])
    forces= np.vstack((data[:, 3],
                       data[:, 6],
                       data[:, 9],
                       data[:, 12])).transpose()
    return time,data,forces

def fix_fz(forces_fz):
    for x in np.nditer(forces_fz, op_flags=['readwrite']):
        if x[...] < 0:
            x[...] = 0.001#######################################################################################
    return forces_fz

def u_friction(data,F_z):
    F_xy = np.zeros((F_z.shape[0], F_z.shape[1]))
    mag = np.zeros((F_z.shape[0], F_z.shape[1]))
    for i in range(0, F_z.shape[1]):
        F_xy[:, i] = np.linalg.norm(data[:, 1 + 3 * i:3 + 3 * i], axis=1)
        mag[:,i] = np.linalg.norm(data[:, 1 + 3 * i:4 + 3 * i], axis=1)
    u = np.divide(F_xy, F_z)
    return u, mag, F_xy


###################################################PARAMETERS###########################################################
'''
what region of data should be used
'''

time_start=0
time_end=-1

fig, axis = plt.subplots(5, sharex=True)

_, data1, forces1 = process_data('datalogs/trot2.txt',time_start,time_end,0)
#_, data2, forces2  = process_data('datalogs/trot3.txt',time_start,time_end,1)
_, data3, forces3  = process_data('datalogs/walk4.txt',time_start,time_end,2)
#_, data4, forces4  = process_data('datalogs/walk5.txt',time_start,time_end,3)
#_, data5, forces5   = process_data('datalogs/walk7.txt',time_start,time_end,4)
#
data=np.vstack((data1,data3))##,data4,data5))###########################################################################################
F_z=np.vstack((forces1,forces3))############,forces4,forces5))#####################################################################################
F_z=fix_fz(F_z)

u_matrix_cof, magnitude,  F_xy =u_friction(data,F_z)
#
for f in np.nditer(u_matrix_cof, op_flags=['readwrite']):
    if (f[...] > 1000):
        f[...] = 1000


################# PLOT U ###############################################################################################

# print(u_matrix_cof.shape)
# fig3d0 = plt.figure()
# plt.scatter(u_matrix_cof[:, 0], u_matrix_cof[:, 1])





#PARAMS FOR CLUSTERING

fxy=np.array(F_xy[:,0])
fz=np.array(F_z[:,0])
fx=np.array(data[:,1])
fy=np.array(data[:,2])
ucof=np.array(u_matrix_cof[:,0])

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']




#ax3d0 = fig3d0.add_subplot(111, projection='3d')
#ax3d0.scatter(X_reduced[:,0],X_reduced[:,1],X_reduced[:,2])
#print(X_reduced)




#########################################################K-MEANS########################################################
# from sklearn.cluster import KMeans
# fig, axe = plt.subplots(2, 2)
#
#

#X=np.vstack((fxy,fz)).transpose()


#
# for ncenters,ax in enumerate(axe.reshape(-1),2):
#     kmeans = KMeans(n_clusters=ncenters, max_iter=10000, tol=0.0005,precompute_distances=True).fit(X)
#     print('centers ' + str(ncenters) + " \n " + str(kmeans.cluster_centers_) )
#     #ax.set_title('centers ' + str(ncenters) )
#     ax.scatter(X[:, 0], X[:, 1],c=kmeans.labels_)
#
#     #plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_
#     # Mark the center of each fuzzy cluster
#     for pt in kmeans.cluster_centers_:
#         ax.plot(pt[0],pt[1], 'rs')

## CLUSTERING FXY vs FZ#################################################################################################

fig1, axes1 = plt.subplots(2, 2)
alldata = np.vstack((fxy, fz))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.0005, maxiter=10000, init=None)
    print('centers ' + str(ncenters) + " \n " + str(cntr) )

    # Store fpc values for later

    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(fxy[cluster_membership == j],
                fz[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))


fig1.tight_layout()


#CLUSTERING u #################################################################################################
#
# figu, axesu = plt.subplots(2, 2)
#
# ucof_0=np.zeros_like(ucof)
# alldata = np.vstack((ucof, ucof_0))
# fpcs = []
#
# for ncenters, ax in enumerate(axesu.reshape(-1), 2):
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         alldata, ncenters, 2, error=0.0005, maxiter=10000, init=None)
#     print('centers ' + str(ncenters) + " \n " + str(cntr) )
#
#     # Store fpc values for later
#
#     fpcs.append(fpc)
#
#     # Plot assigned clusters, for each data point in training set
#     cluster_membership = np.argmax(u, axis=0)
#     for j in range(ncenters):
#         ax.plot(ucof[cluster_membership == j],
#                 ucof_0[cluster_membership == j], '.', color=colors[j])
#
#     # Mark the center of each fuzzy cluster
#     for pt in cntr:
#         ax.plot(pt[0], pt[1], 'rs')
#
#     ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
#
#
# figu.tight_layout()

###### FX FY FZ ########################################################################################################
#
# alldata = np.vstack((fx,fy,fz))
# print(alldata.shape)
# plt.show()
# fpcs = []
#
# for ncenters in np.arange(2,6):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         alldata, ncenters, 2, error=0.00005, maxiter=100000, init=None)
#     print('centers ' + str(ncenters) + " \n " + str(cntr) )
#
#     # Store fpc values for later
#
#     fpcs.append(fpc)
#
#     # Plot assigned clusters, for each data point in training set
#     cluster_membership = np.argmax(u, axis=0)
#     print(cluster_membership)
#     for j in range(ncenters):
#         ax.scatter(fx[cluster_membership == j],
#                 fy[cluster_membership == j],fz[cluster_membership == j], '.', color=colors[j])
#
#     # Mark the center of each fuzzy cluster
#     for pt in cntr:
#         ax.scatter(pt[0], pt[1],pt[2], 'rs')
#
#     ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
#
#




# fig = plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(u_cof[:,0], np.zeros_like(u_cof[:,0]), 'o')
# plt.ylim((-0.1,0.1))
# plt.xlim((0,100))
#
#
#
# fig3d0 = plt.figure()
# ax3d0 = fig3d0.add_subplot(111, projection='3d')
# ax3d0.scatter(data[:,1],data[:,2],data[:,3],c=u_cof[:,0])
#
# ax3d0.set_xlabel('Fx Label')
# ax3d0.set_ylabel('Fy Label')
# ax3d0.set_zlabel('Fz Label')
# plt.title('Forces with u coff')
#
# fig3d1 = plt.figure()
# ax3d1 = fig3d1.add_subplot(111, projection='3d')
# ax3d1.scatter(data[:,1],data[:,2],data[:,3],c=magnitude[:,0])
# ax3d1.set_xlabel('Fx Label')
# ax3d1.set_ylabel('Fy Label')
# ax3d1.set_zlabel('Fz Label')
# plt.title('Forces with magnitude')

plt.show()

