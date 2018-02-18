import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from mpl_toolkits.mplot3d import Axes3D
import subprocess,sys


pd.set_option('expand_frame_repr', False)
plt.close("all")
#####################################################FUNCTIONS##########################################################
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

#####################################################PARAMETERS#########################################################
data_load = [True,
             False,
             True,
             False,
             False,
             ]
text = ['datalogs/trot2.txt',
        'datalogs/trot3.txt',
        'datalogs/walk4.txt',
        'datalogs/walk5.txt',
        'datalogs/walk7.txt']
mass_hyq=850
time_start = 0
time_end = 900
distance_drop=1

frames_ = []
force1leg_ = []
force4legs_ = []

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

########################################READING DATA####################################################################

for i, load in enumerate(data_load):
    if load == True:
        datalog = pd.read_csv(text[i], sep='\t', names=['TIME',
                                                        'LEG1_fx', 'LEG1_fy', 'LEG1_fz',
                                                        'LEG2_fx', 'LEG2_fy', 'LEG2_fz',
                                                        'LEG3_fx', 'LEG3_fy', 'LEG3_fz',
                                                        'LEG4_fx', 'LEG4_fy', 'LEG4_fz'])
        print('Data: ', text[i], 'loaded ', datalog.shape)
        a = datalog[datalog['TIME'] >= time_start].index[0]
        b = datalog[datalog['TIME'] <= time_end].index[-1]
        datalog = datalog[a:b + 1]
        #print('Data: ', text[i], ' cut ', '\033[1;32m',datalog.shape,'\033[1;m' , ' from ', time_start, ' to ', time_end)
        datalog.loc[datalog.LEG1_fz < 0, 'LEG1_fz'] = 0.0001;
        datalog.loc[datalog.LEG2_fz < 0, 'LEG2_fz'] = 0.0001;
        datalog.loc[datalog.LEG3_fz < 0, 'LEG3_fz'] = 0.0001;
        datalog.loc[datalog.LEG4_fz < 0, 'LEG4_fz'] = 0.0001;
       # print('Fixing Fz issue...')
        if False:
            print('Plotting forces in 1 leg in x,y,z direction...')
            fig_fxyz, axis_xyz = plt.subplots(3, sharex=True)
            plt.title('Forces in 1 leg in x,y,z direction ' + text[i])
            datalog.plot(ax=axis_xyz[0], x='TIME', y='LEG1_fx')
            datalog.plot(ax=axis_xyz[1], x='TIME', y='LEG1_fy')
            datalog.plot(ax=axis_xyz[2], x='TIME', y='LEG1_fz')
            if False:
                print('Plotting dots to visualize real data for one leg...')
                datalog.plot.scatter(ax=axis_xyz[0], x='TIME', y='LEG1_fx',c='r')
                datalog.plot.scatter(ax=axis_xyz[1], x='TIME', y='LEG1_fy',c='r')
                datalog.plot.scatter(ax=axis_xyz[2], x='TIME', y='LEG1_fz',c='r')

        if False:
            fig_z, axis_z = plt.subplots(4, sharex=True)
            plt.title('Plotting forces in 4 legs in z direction' + text[i])
            datalog.plot(ax=axis_z[0], x='TIME', y='LEG1_fz')
            datalog.plot(ax=axis_z[1], x='TIME', y='LEG2_fz')
            datalog.plot(ax=axis_z[2], x='TIME', y='LEG3_fz')
            datalog.plot(ax=axis_z[3], x='TIME', y='LEG4_fz')
            if False:
                print('Plotting dots to visualize real data for one leg...')
                datalog.plot.scatter(ax=axis_z[0], x='TIME', y='LEG1_fz',c='r')
                datalog.plot.scatter(ax=axis_z[1], x='TIME', y='LEG2_fz',c='r')
                datalog.plot.scatter(ax=axis_z[2], x='TIME', y='LEG3_fz',c='r')
                datalog.plot.scatter(ax=axis_z[3], x='TIME', y='LEG4_fz', c='r')

        #on one plot
        if False:
            fig_whole, axis_whole = plt.subplots(1, sharex=True)
            print('Everything on one plot z components...')
            datalog.plot(ax=axis_whole, x='TIME', y='LEG1_fz', c='b')
            datalog.plot(ax=axis_whole, x='TIME', y='LEG2_fz', c='orange')
            datalog.plot(ax=axis_whole, x='TIME', y='LEG3_fz', c='g')
            datalog.plot(ax=axis_whole, x='TIME', y='LEG4_fz', c='c')



        F_sum = pd.DataFrame({
            'F_sum': datalog['LEG1_fz'] + datalog['LEG2_fz'] + datalog['LEG3_fz'] + datalog['LEG4_fz']
        })
        datalog=datalog.join(F_sum)
        #print('Calculation F_sum of forces in z direction ', F_sum.shape, ' addint to datalog', datalog.shape)
        if False:
            print('Plotting F_sum over time')
            plt.figure()
            plt.title('Max Sum')
            plt.plot(datalog['TIME'],datalog['F_sum'])
            #print('Average',datalog['F_sum'].mean())
            if False:
                print('Plotting dots to visualize real data for F_sum...')
                plt.scatter(datalog['TIME'],F_sum, c='r')

        frames_.append(datalog)
        force1leg_.append(datalog[['LEG1_fx', 'LEG1_fy', 'LEG1_fz']])

        force4legs_.append(datalog[['LEG1_fx', 'LEG1_fy', 'LEG1_fz',
                                'LEG2_fx', 'LEG2_fy', 'LEG2_fz',
                                'LEG3_fx', 'LEG3_fy', 'LEG3_fz',
                                'LEG4_fx', 'LEG4_fy', 'LEG4_fz']])




###############################################CALCULATION & REARRANGEMENT #############################################

data = pd.concat(frames_)  ##KEEP IN MIND THAT WE LOSE ORDER OF TIME, BECAUSE DATA IS STACKED!!!!!!!!!!!!!!!!!!!!!!!!!!!
force1leg = pd.concat(force1leg_)
force4legs = pd.concat(force4legs_)

#print()
#print('GATHERING DATA', '\033[1;32m',data.shape,'\033[1;m')
#print(data.head(1))
#print(data.tail(1))

# ###############################################CUT WRONG DATA THANKS TO CMEAN AND MASS##################################


if False:
    number_of_centers_Fsum=3
    print('\033[1;33mRejecting corrupted data based on F_sum\033[1;m')
    likefsum=np.zeros_like(data['F_sum'])
    alldata_fsum = np.vstack((data['F_sum'], likefsum))
    fpcs = []
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata_fsum, number_of_centers_Fsum, 2, error=0.0005, maxiter=10000, init=None)
    print('centers '+ str(cntr) )

    if False:
        print('Drawing distribution of rejecting data, number of clusters ', number_of_centers_Fsum )
        fig2, ax2 = plt.subplots()
        ax2.set_title('Trained model')
        for j in range(3):
            ax2.plot(alldata_fsum[0, u_orig.argmax(axis=0) == j],
                     alldata_fsum[1, u_orig.argmax(axis=0) == j], 'o',
                    label='series ' + str(j))
        ax2.legend()

    proper_mass_index=np.where(cntr[:,0]==np.median(cntr[:,0]))[0]
    print(proper_mass_index)
    data = data.drop(data[np.invert(u_orig.argmax(axis=0) == proper_mass_index)].index)
   # force1leg = pd.concat(force1leg_)
    #force4legs = pd.concat(force4legs_)
    force1leg=force1leg.drop(force1leg[np.invert(u_orig.argmax(axis=0) == proper_mass_index)].index)
    force4legs=force4legs.drop(force4legs[np.invert(u_orig.argmax(axis=0) == proper_mass_index)].index)
    print('Data cutting sum ',data.shape,' 1 leg ' , force1leg.shape, ' force 4 legs ' , force4legs.shape)
    if False:
        print('F_sum over time after lifting')
        plt.figure()
        plt.title('After lifiting')
        plt.plot(data['TIME'], data['F_sum'])

elif False:
  #  print()
    print('\033[1;33m rejecting data based on distance\033[1;m')
    print(data.shape)
    Distance = pd.DataFrame({'dist': np.divide(np.abs(-mass_hyq + data['LEG1_fz'] + data['LEG2_fz'] + data['LEG3_fz'] + data['LEG4_fz']), np.sqrt(4))
                             })
    data = data.drop(data[Distance['dist'] > distance_drop].index)
    force1leg = force1leg.drop(force1leg[Distance['dist'] > distance_drop].index)
    force4legs = force4legs.drop(force4legs[Distance['dist'] > distance_drop].index)
    #print('Data based on distance  ', data.shape, ' 1 leg ', force1leg.shape, ' force 4 legs ', force4legs.shape)
    if False:
        print('F_sum over time after lifting')
        plt.figure()
        plt.title('After lifiting')
        plt.plot(data['TIME'], data['F_sum'])


else:
   # print()
    print('\033[1;33mWithout rejecting data\033[1;m')
    #force1leg = pd.concat(force1leg_)
    #force4legs = pd.concat(force4legs_)





 ########################################################################################################################

#print()
#print('Data from all legs concatenated', '\033[1;32m',force4legs.shape,'\033[1;m')


F_xy = pd.DataFrame({'LEG1_xy': force4legs[['LEG1_fx', 'LEG1_fy']].apply(np.linalg.norm, axis=1),
                     'LEG2_xy': force4legs[['LEG2_fx', 'LEG2_fy']].apply(np.linalg.norm, axis=1),
                     'LEG3_xy': force4legs[['LEG3_fx', 'LEG3_fy']].apply(np.linalg.norm, axis=1),
                     'LEG4_xy': force4legs[['LEG4_fx', 'LEG4_fy']].apply(np.linalg.norm, axis=1),
                     })

#print()
print('Lateral forces calculated', '\033[1;32m',F_xy.shape,'\033[1;m')
#print(F_xy.head(2))

F_abs = pd.DataFrame({'LEG1_abs': force4legs[['LEG1_fx', 'LEG1_fy', 'LEG1_fz']].apply(np.linalg.norm, axis=1),
                      'LEG2_abs': force4legs[['LEG2_fx', 'LEG2_fy', 'LEG2_fz']].apply(np.linalg.norm, axis=1),
                      'LEG3_abs': force4legs[['LEG3_fx', 'LEG3_fy', 'LEG3_fz']].apply(np.linalg.norm, axis=1),
                      'LEG4_abs': force4legs[['LEG4_fx', 'LEG4_fy', 'LEG4_fz']].apply(np.linalg.norm, axis=1),
                      })
#print()
print('ABS forces calculated', '\033[1;32m',F_abs.shape,'\033[1;m')
#print(F_abs.head(2))

friction_coefficient = pd.DataFrame({'LEG1_u': np.divide(F_xy['LEG1_xy'], force4legs['LEG1_fz']),
                                     'LEG2_u': np.divide(F_xy['LEG2_xy'], force4legs['LEG2_fz']),
                                     'LEG3_u': np.divide(F_xy['LEG3_xy'], force4legs['LEG3_fz']),
                                     'LEG4_u': np.divide(F_xy['LEG4_xy'], force4legs['LEG4_fz']),
                                     })

if True:
    bound = 100
    #print()
    #print('Calkculation of friction coefficient with bound restriction to ', '\033[1;31m',bound,'\033[1;m' )
    friction_coefficient.loc[friction_coefficient.LEG1_u > bound, 'LEG1_u'] = bound;
    friction_coefficient.loc[friction_coefficient.LEG2_u > bound, 'LEG2_u'] = bound;
    friction_coefficient.loc[friction_coefficient.LEG3_u > bound, 'LEG3_u'] = bound;
    friction_coefficient.loc[friction_coefficient.LEG4_u > bound, 'LEG4_u'] = bound;

#print()
print('Friction coefficient forces calculated', '\033[1;32m',friction_coefficient.shape,'\033[1;m')

#print(friction_coefficient.head(2))

# #
# #
# #
# # ###################ANDRZEJ######################
# #
friction_andrzej = pd.DataFrame({'LEG1_u': np.divide(force4legs['LEG1_fz'], F_abs['LEG1_abs']),
                                 'LEG2_u': np.divide(force4legs['LEG2_fz'], F_abs['LEG2_abs']),
                                 'LEG3_u': np.divide(force4legs['LEG3_fz'], F_abs['LEG3_abs']),
                                 'LEG4_u': np.divide(force4legs['LEG4_fz'], F_abs['LEG4_abs']),
                                 })

#print()
#print('andrzej', friction_andrzej.shape)
#print(friction_andrzej.head(2))

if True:
    fig_u_2D = plt.figure()
    plt.xlabel('LEG1_u')
    plt.ylabel('LEG2_u')
    plt.title('friction_andrzej leg1vsleg2')
    plt.scatter(friction_andrzej['LEG1_u'], friction_andrzej['LEG2_u'])

    fig_u_2D = plt.figure()
    plt.xlabel('LEG1_u')
    plt.ylabel('LEG3_u')
    plt.title('friction_andrzej leg1vsleg3')
    plt.scatter(friction_andrzej['LEG1_u'], friction_andrzej['LEG3_u'])

    fig_u_2D = plt.figure()
    plt.xlabel('LEG1_u')
    plt.ylabel('LEG4_u')
    plt.title('friction_andrzej leg1vsleg4')
    plt.scatter(friction_andrzej['LEG1_u'], friction_andrzej['LEG4_u'])

if True:
    fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(111, projection='3d')
    ax_forces_3d.set_xlabel('friction_andrzej F1z')
    ax_forces_3d.set_ylabel('friction_andrzej F2z')
    ax_forces_3d.set_zlabel('friction_andrzej F3z')
    plt.title('friction_andrzej comparison')
    ax_forces_3d.scatter(friction_andrzej['LEG2_u'], friction_andrzej['LEG3_u'], friction_andrzej['LEG4_u'],c=friction_andrzej['LEG1_u'])


if False:
    fig_u_1D = plt.figure()
    plt.xlabel('LEG_u')
    plt.title('coefficients from one leg')
    plt.scatter(friction_andrzej['LEG1_u'], np.zeros_like(friction_andrzej['LEG1_u']))

if False:
    #fig.figure()
    plt.xlabel('time')
    plt.ylabel('LEG1_u')
    plt.title('how friction coefficion changes')
    plt.scatter(data['TIME'], 300*friction_coefficient['LEG1_u'])
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('LEG1_u')
    plt.title('how friction coefficion changes')
    plt.scatter(data['TIME'], friction_coefficient['LEG1_u'])


# # # ############################################VISUALISATION###############################################################
#print()
#print('Data Visualisation')

if False:
    fig_u_2D = plt.figure()
    plt.subplot(221)
    plt.xlabel('LEG1_u')
    plt.ylabel('LEG2_u')
    plt.title('friction_coefficient leg1 vs leg2')
    plt.scatter(friction_coefficient['LEG1_u'], friction_coefficient['LEG2_u'])

    #fig_u_2D = plt.figure()
    plt.subplot(222)
    plt.xlabel('LEG1_u')
    plt.ylabel('LEG3_u')
    plt.title('friction_coefficient leg1 vs leg3')
    plt.scatter(friction_coefficient['LEG1_u'], friction_coefficient['LEG3_u'])

    #fig_u_2D = plt.figure()
    plt.subplot(223)
    plt.xlabel('LEG1_u')
    plt.ylabel('LEG4_u')
    plt.title('friction_coefficient leg1 vs leg4')
    plt.scatter(friction_coefficient['LEG1_u'], friction_coefficient['LEG4_u'])


if False:
    fig_u_2D = plt.figure()
    plt.subplot(221)
    plt.xlabel('LEG1_u')
    plt.ylabel('F_abs')
    plt.title('F_abs vs Fu')
    plt.scatter(friction_coefficient['LEG1_u'], F_abs['LEG1_abs'])



if False:
    fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(111, projection='3d')
    ax_forces_3d.set_xlabel('Ux')
    ax_forces_3d.set_ylabel('Uy')
    ax_forces_3d.set_zlabel('Uz')
    plt.title('U')
    ax_forces_3d.scatter(friction_coefficient['LEG1_u'], friction_coefficient['LEG2_u'], friction_coefficient['LEG3_u'])

if False:
    fig_u_1D = plt.figure()
    plt.subplot(221)
    plt.xlabel('LEG1_u')
    plt.title('coefficient from leg1')
    plt.scatter(friction_coefficient['LEG1_u'], np.zeros_like(friction_coefficient['LEG1_u']))

    #fig_u_1D = plt.figure()
    plt.subplot(222)
    plt.xlabel('LEG2_u')
    plt.title('oefficient from leg2')
    plt.scatter(friction_coefficient['LEG2_u'], np.zeros_like(friction_coefficient['LEG2_u']))

    #fig_u_1D = plt.figure()
    plt.subplot(223)
    plt.xlabel('LEG3_u')
    plt.title('coefficient from leg3')
    plt.scatter(friction_coefficient['LEG3_u'], np.zeros_like(friction_coefficient['LEG3_u']))


    #fig_u_1D = plt.figure()
    plt.subplot(224)
    plt.xlabel('LEG4_u')
    plt.title('coefficient from leg4')
    plt.scatter(friction_coefficient['LEG4_u'], np.zeros_like(friction_coefficient['LEG4_u']))

if False:
    fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(111, projection='3d')
    ax_forces_3d.set_xlabel('Fx')
    ax_forces_3d.set_ylabel('Fy')
    ax_forces_3d.set_zlabel('Fz')
    plt.title('fx,fy,fz in leg1')
    ax_forces_3d.scatter(force4legs['LEG1_fx'], force4legs['LEG1_fy'], force4legs['LEG1_fz'])

    #fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(121, projection='3d')
    ax_forces_3d.set_xlabel('Fx')
    ax_forces_3d.set_ylabel('Fy')
    ax_forces_3d.set_zlabel('Fz')
    plt.title('fx,fy,fz in leg2')
    ax_forces_3d.scatter(force4legs['LEG2_fx'], force4legs['LEG2_fy'], force4legs['LEG2_fz'])

    #fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(211, projection='3d')
    ax_forces_3d.set_xlabel('Fx')
    ax_forces_3d.set_ylabel('Fy')
    ax_forces_3d.set_zlabel('Fz')
    plt.title('fx,fy,fz in leg3')
    ax_forces_3d.scatter(force4legs['LEG3_fx'], force4legs['LEG3_fy'], force4legs['LEG3_fz'])

    #fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(221, projection='3d')
    ax_forces_3d.set_xlabel('Fx')
    ax_forces_3d.set_ylabel('Fy')
    ax_forces_3d.set_zlabel('Fz')
    plt.title('fx,fy,fz in leg4')
    ax_forces_3d.scatter(force4legs['LEG4_fx'], force4legs['LEG4_fy'], force4legs['LEG4_fz'])


if False:
    fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(111, projection='3d')
    ax_forces_3d.set_xlabel('Fx')
    ax_forces_3d.set_ylabel('Fy')
    ax_forces_3d.set_zlabel('Fz')
    plt.title('forces in one leg')
    ax_forces_3d.scatter(force4legs['LEG1_fx'], force4legs['LEG1_fy'], force4legs['LEG1_fz'], c=F_abs['LEG1_abs'])

if False:
    fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(111, projection='3d')
    ax_forces_3d.set_xlabel('Fx')
    ax_forces_3d.set_ylabel('Fy')
    ax_forces_3d.set_zlabel('Fz')
    plt.title('forces in one leg')
    ax_forces_3d.scatter(force4legs['LEG1_fx'], force4legs['LEG1_fy'], force4legs['LEG1_fz'],
                         c=friction_coefficient['LEG1_u'])

if False:
    fig_forces_3D = plt.figure()
    ax_forces_3d = fig_forces_3D.add_subplot(111, projection='3d')
    ax_forces_3d.set_xlabel('F1z')
    ax_forces_3d.set_ylabel('F2z')
    ax_forces_3d.set_zlabel('F3z')
    plt.title('forces in one leg')
    ax_forces_3d.scatter(force4legs['LEG1_fz'], force4legs['LEG2_fz'], force4legs['LEG3_fz'],c=force4legs['LEG4_fz'])

# FZ
if False:
    fig_u_2D = plt.figure()
    plt.subplot(221)
    plt.xlabel('LEG1_fz')
    plt.ylabel('LEG2_fz')
    plt.title('Fz leg 1 vs 2')
    plt.scatter(force4legs['LEG1_fz'], force4legs['LEG2_fz'])

    plt.subplot(222)
    #fig_u_2D = plt.figure()
    plt.xlabel('LEG1_fz')
    plt.ylabel('LEG3_fz')
    plt.title('Fz leg 1 vs 3')
    plt.scatter(force4legs['LEG1_fz'], force4legs['LEG3_fz'])

    plt.subplot(223)
    #fig_u_2D = plt.figure()
    plt.xlabel('LEG1_fz')
    plt.ylabel('LEG4_fz')
    plt.title('Fz leg 1 vs 4')
    plt.scatter(force4legs['LEG1_fz'], force4legs['LEG4_fz'])




#
# # ################################################CLUSTERING##############################################################
# # ########################################################################################################################
# # ###############################################CMEANS###################################################################
#
if True:
    #print()
    print('C-MEANS CLUSTERING')
    F_xy_stack=np.hstack((F_xy['LEG1_xy'],F_xy['LEG2_xy'],F_xy['LEG3_xy'],F_xy['LEG4_xy']))
    F_z_stack=np.hstack((force4legs['LEG1_fz'],force4legs['LEG2_fz'],force4legs['LEG3_fz'],force4legs['LEG4_fz']))
    #print(F_z_stack.shape)
    if False:
        NHA=2
        F_z_stack=running_mean(np.hstack((np.ones(NHA-1),F_z_stack)),NHA)
        print('RUNNING MEAN')
    alldata = np.vstack((F_xy_stack, F_z_stack))

    #print(F_z_stack.shape)


    #print(alldata.shape)
if True:
    fpcs = []
    fig, axis = plt.subplots(2, 2)
    axis = axis.reshape(-1)
    for i, ncenters in enumerate(np.arange(2, 6)):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.0005, maxiter=10000, init=None)
        print('centers ' + str(ncenters) + " \n " + str(cntr))
        fpcs.append(fpc)
        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        # print(F_xy.loc[cluster_membership == 0, 'LEG1_xy'])
        for j in range(ncenters):
            axis[i].plot(F_xy_stack[cluster_membership == j], F_z_stack[cluster_membership == j], '.', color=colors[j])
        #
        #       Mark the center of each fuzzy cluster
        for pt in cntr:
            axis[i].plot(pt[0], pt[1], 'rs')

        axis[i].set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    fig.tight_layout()

# ###############################################KMEANS###################################################################

# if False:
#     print()
#     print('K-MEANS Clustering')
#     from sklearn.cluster import KMeans
#
#     fig, axe = plt.subplots(2, 2)
#     alldata = np.vstack((F_xy['LEG1_xy'], force4legs['LEG1_fz'])).transpose()
#
#     for ncenters, ax in enumerate(axe.reshape(-1), 2):
#         kmeans = KMeans(n_clusters=ncenters, max_iter=10000, tol=0.0005, precompute_distances=True).fit(alldata)
#         print('centers ' + str(ncenters) + " \n " + str(kmeans.cluster_centers_))
#         # ax.set_title('centers ' + str(ncenters) )
#         ax.scatter(alldata[:, 0], alldata[:, 1], c=kmeans.labels_)
#         # Mark the center of each fuzzy cluster
#         for pt in kmeans.cluster_centers_:
#             ax.plot(pt[0], pt[1], 'rs')
#


# ######################################PREDICTION################################################################################
# number_of_clusters=2
#
# NEWcntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
#     alldata, number_of_clusters, 2, error=0.005, maxiter=1000)
#
# print("THE LAST CLUSTER IS",NEWcntr)
#
#
# x= np.linspace(0, 600, 1000)
# z= 200*np.sign(x)
#
# axis_whole.plot(x,z,'r')


###############################################COUPLING LEGS############################################################
#
#
# F_HYQ=850
#
# N_DIM_CLUSTERS=np.array([[F_HYQ,0,0,0],  #1 (0)
#                         [0,F_HYQ,0,0],   #2 (1)
#                         [0,0,F_HYQ,0],   #3 (2)
#                         [0,0,0,F_HYQ],   #4 (4)
#
#                         [F_HYQ/2,F_HYQ/2,0,0],   #12 (4)
#                         [F_HYQ/2,0,F_HYQ/2,0],   #13 (5)
#                         [F_HYQ/2,0,0,F_HYQ/2],   #14 (6)
#                         [0,F_HYQ/2,F_HYQ/2,0],   #23 (7)
#                         [0,F_HYQ/2,0,F_HYQ/2],   #24 (8)
#                         [0,0,F_HYQ/2,F_HYQ/2],   #34 (9)
#
#                         [F_HYQ/3,F_HYQ/3,F_HYQ/3,0],   #123 (10)
#                         [F_HYQ/3,F_HYQ/3,0,F_HYQ/3],   #124 (11)
#                         [0,F_HYQ/3,F_HYQ/3,F_HYQ/3],   #234 (12)
#                         [F_HYQ/3,0,F_HYQ/3,F_HYQ/3],   #134 (13)
#
#                         [F_HYQ/4,F_HYQ/4,F_HYQ/4,F_HYQ/4],   #1234 (14)
#                         ])
#
#
# attempts=np.vstack((force4legs['LEG1_fz'],force4legs['LEG2_fz'],force4legs['LEG3_fz'],force4legs['LEG4_fz']))
# distance_patrition=np.zeros((15,attempts.shape[1]))
#
# for i in np.arange(attempts.shape[1]):
#     distance_patrition[:,i]=np.linalg.norm((N_DIM_CLUSTERS - attempts[:,i]),axis=1)
#
# sum_dist=np.sum(distance_patrition,axis=0)
# u_partition=distance_patrition/sum_dist
#
# check_sum=np.sum(u_partition,axis=0)
#
# state=u_partition.argmin(axis=0)
#
# #print(state.shape)
#
#
#
# #print(state.shape)
# #print(datalog['TIME'].shape)


#plt.scatter(datalog['TIME'],100*state,c='r')#datalog['TIME'],state,'r')
#plt.grid('on')
#print(z)

#plt.scatter(datalog['TIME'],F_abs['LEG1_abs'],c='r')#datalog['TIME'],state,'r')


plt.show()