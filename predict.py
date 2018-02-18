import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('expand_frame_repr', False)
plt.close("all")
#####################################################FUNCTIONS##########################################################


#####################################################PARAMETERS#########################################################
data_load = [False,
             True,
             False,
             False,
             False]
text = ['datalogs/trot2.txt',
        'datalogs/trot3.txt',
        'datalogs/walk4.txt',
        'datalogs/walk5.txt',
        'datalogs/walk7.txt']

time_start = 0
time_end = 300

frames_ = []
force1leg_ = []
force4legs_ = []

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

NEWcntr=np.array([[ 79.90644043, 394.72497026],
 [ 25.8244616,   49.10059144]])
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
        #FIXING fz<0
        datalog.loc[datalog.LEG1_fz < 0, 'LEG1_fz'] = 0.0001;
        datalog.loc[datalog.LEG2_fz < 0, 'LEG2_fz'] = 0.0001;
        datalog.loc[datalog.LEG3_fz < 0, 'LEG3_fz'] = 0.0001;
        datalog.loc[datalog.LEG4_fz < 0, 'LEG4_fz'] = 0.0001;

        if False:
            fig, axis = plt.subplots(3, sharex=True)
            plt.title('Distribution of forces in 1 LEG in ' + text[i])
            if False:   #dots to visualize
                datalog.plot.scatter(ax=axis[0], x='TIME', y='LEG1_fx',c='r')
                datalog.plot.scatter(ax=axis[1], x='TIME', y='LEG1_fy',c='r')
                datalog.plot.scatter(ax=axis[2], x='TIME', y='LEG1_fz',c='r')
            datalog.plot(ax=axis[0], x='TIME', y='LEG1_fx')
            datalog.plot(ax=axis[1], x='TIME', y='LEG1_fy')
            datalog.plot(ax=axis[2], x='TIME', y='LEG1_fz')

        if False:
            fig, axis = plt.subplots(4, sharex=True)
            plt.title('Distribution of Fz in each leg' + text[i])
            datalog.plot(ax=axis[0], x='TIME', y='LEG1_fz')
            datalog.plot(ax=axis[1], x='TIME', y='LEG2_fz')
            datalog.plot(ax=axis[2], x='TIME', y='LEG3_fz')
            datalog.plot(ax=axis[3], x='TIME', y='LEG4_fz')

        F_sum = pd.DataFrame({
            'F_sum': datalog['LEG1_fz'] + datalog['LEG2_fz'] + datalog['LEG3_fz'] + datalog['LEG4_fz']
        })
        datalog=datalog.join(F_sum)

        if False:
            plt.figure()
            plt.title('Max Sum')
            plt.plot(datalog['TIME'],datalog['F_sum'])
            if False:
                plt.scatter(datalog['TIME'],F_sum, c='r')

        ## GATHERING DATA FROM ALL PLOTS
        frames_.append(datalog)
        force1leg_.append(datalog[['LEG1_fx', 'LEG1_fy', 'LEG1_fz']])

        force4legs_.append(datalog[['LEG1_fx', 'LEG1_fy', 'LEG1_fz',
                                'LEG2_fx', 'LEG2_fy', 'LEG2_fz',
                                'LEG3_fx', 'LEG3_fy', 'LEG3_fz',
                                'LEG4_fx', 'LEG4_fy', 'LEG4_fz']])




###############################################CALCULATION & REARRANGEMENT #############################################

data = pd.concat(frames_)
print()
print('data from ', time_start, 's to ', time_end, 's ', data.shape)
print(data.head(1))
#print(data.tail(1))

# ###############################################CUT WRONG DATA THANKS TO CMEAN AND MASS##################################


force1leg = pd.concat(force1leg_)
force4legs = pd.concat(force4legs_)


# ########################################################################################################################

print()
print('Data from all legs concatenated', force4legs.shape)


F_xy = pd.DataFrame({'LEG1_xy': force4legs[['LEG1_fx', 'LEG1_fy']].apply(np.linalg.norm, axis=1),
                     'LEG2_xy': force4legs[['LEG2_fx', 'LEG2_fy']].apply(np.linalg.norm, axis=1),
                     'LEG3_xy': force4legs[['LEG3_fx', 'LEG3_fy']].apply(np.linalg.norm, axis=1),
                     'LEG4_xy': force4legs[['LEG4_fx', 'LEG4_fy']].apply(np.linalg.norm, axis=1),
                     })

print()
print('Lateral forces calculated', F_xy.shape)
print(F_xy.head(2))

F_abs = pd.DataFrame({'LEG1_abs': force4legs[['LEG1_fx', 'LEG1_fy', 'LEG1_fz']].apply(np.linalg.norm, axis=1),
                      'LEG2_abs': force4legs[['LEG2_fx', 'LEG2_fy', 'LEG2_fz']].apply(np.linalg.norm, axis=1),
                      'LEG3_abs': force4legs[['LEG3_fx', 'LEG3_fy', 'LEG3_fz']].apply(np.linalg.norm, axis=1),
                      'LEG4_abs': force4legs[['LEG4_fx', 'LEG4_fy', 'LEG4_fz']].apply(np.linalg.norm, axis=1),
                      })
print()
print('ABS forces calculated', F_abs.shape)
print(F_abs.head(2))

friction_coefficient = pd.DataFrame({'LEG1_u': np.divide(F_xy['LEG1_xy'], force4legs['LEG1_fz']),
                                     'LEG2_u': np.divide(F_xy['LEG2_xy'], force4legs['LEG2_fz']),
                                     'LEG3_u': np.divide(F_xy['LEG3_xy'], force4legs['LEG3_fz']),
                                     'LEG4_u': np.divide(F_xy['LEG4_xy'], force4legs['LEG4_fz']),
                                     })

if True:
    bound = 500
    friction_coefficient.loc[friction_coefficient.LEG1_u > bound, 'LEG1_u'] = bound;
    friction_coefficient.loc[friction_coefficient.LEG2_u > bound, 'LEG2_u'] = bound;
    friction_coefficient.loc[friction_coefficient.LEG3_u > bound, 'LEG3_u'] = bound;
    friction_coefficient.loc[friction_coefficient.LEG4_u > bound, 'LEG4_u'] = bound;

print()
print('Friction coefficient forces calculated', friction_coefficient.shape)
print(friction_coefficient.head(2))

######################################PREDICTION################################################################################




newdata= np.vstack((F_xy['LEG1_xy'], force4legs['LEG1_fz'])).transpose()

u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, NEWcntr, 2, error=0.005, maxiter=1000)
212

cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization
print(cluster_membership.shape)

fig3, ax3 = plt.subplots()
ax3.set_title('Random points classifed according to known centers')
for j in range(len(NEWcntr)):
    ax3.plot(newdata[cluster_membership == j, 0],
             newdata[cluster_membership == j, 1], 'o',
             label='series ' + str(j))
ax3.legend()

plt.figure()
plt.title('Predicted')
plt.plot(data['TIME'], u[1,:])
# if False:
#     plt.scatter(datalog['TIME'], F_sum, c='r')
#


plt.show()