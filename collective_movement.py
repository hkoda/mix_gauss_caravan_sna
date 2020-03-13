# coding:utf-8
# python code of the simulation of the social network analysis, community clustering, collective movement generation.
# for the details of the methods, descriptions and research purpose, see
# Koda and Matsuda 2020 in prep or bioRxvs (DOI will appear soon.)
# requirements
# Python 3.6.9 :: Anaconda, Inc.
# numpy                     1.17.4           py36h890c691_0 
# matplotlib                3.1.1            py36h54f8f79_0
# scipy                     1.3.2            py36h1410ff5_0 
# networkx                  2.4                        py_0  
# python-louvain            0.13                       py_0  

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import community
import itertools
import sys
import os

# methods for making a spatial distribution of the multi agents from 
# the bivaiate mixture Gaussian distributions
# core_s is the n_{I}, independenter numbers
# sub_s is the n_{D}, dependener numbers
# r is the ratios of the \sigma_{I} over \sigma_{D}. Given \sigma_{D} always set to 1, r is \sigma_{I} in this simulations.
# agent_s is the 2D coordinations of the multi agent set.
def mk_agent(n_core,n_sub,r):
    core_s = np.random.multivariate_normal([0,0],cov = [[r,0],[0,r]],size = n_core)
    sub_s = np.zeros(n_core * n_sub * 2).reshape(n_core,n_sub,2)
    for i in range(n_core):
        for j in range(n_sub):
            sub_s[i, j,:] = np.random.multivariate_normal(core_s[i],cov = [[1,0],[0,1]],size=1)
    index_list = np.arange(n_core + n_core * n_sub).reshape(-1,1)
    agent_s = np.concatenate([core_s,sub_s.reshape(-1,2)],axis=0)
    agent_s = np.concatenate([index_list,agent_s],axis=1)
    return agent_s

# methods for generating the caravan data by the collective movement rules.
# returns the caraven order sets, from the agent_s.
def serial_move_agent(agent_s):
    move_agent_order = []
    dist_matrix = distance.cdist(agent_s[:,1:],agent_s[:,1:])
    agent_id = np.random.choice(range(len(agent_s)),size = 1)
    for i in range(len(agent_s)):
        move_agent_order.append(agent_id)
        next_id = dist_matrix[agent_id,:].argsort().flatten()[1]
        dist_matrix[agent_id,:] = np.nan
        dist_matrix[:,agent_id] = np.nan
        agent_id = next_id
    return move_agent_order

# loops of the caravan observations.
# experiment_n is the experiment numbers of the caravan observations.
# finally, this returns 
# - agent_s_s: experiment numbers of the agent_s
# - move_agent_observation_dist_matrix: adjacency matrix generated from the experiment numbers of the caravan observations.
# - move_agent_observation: the experiment numbers of the caravan observations
def perform_observation(core_n,sub_n,experiment_n,r):
    agent_s_s = np.zeros(((sub_n + 1) * core_n,3,experiment_n))
    move_agent_observation = np.zeros(experiment_n * (core_n + core_n * sub_n)).reshape(experiment_n,-1)
    move_agent_observation_dist_matrix = np.zeros((experiment_n,(core_n + core_n * sub_n),(core_n + core_n * sub_n)))
    for i in range(experiment_n):
        agent_s = mk_agent(core_n,sub_n,r)
        move_agent_observation[i,:] = serial_move_agent(agent_s)
        agent_s_s[:,:,i] = agent_s
    move_agent_observation_argsort = move_agent_observation[:,:].argsort()
    for i in range(experiment_n):
        move_agent_observation_pseudocoordinate = np.hstack(
            (
                move_agent_observation_argsort[i].reshape(-1,1),
                np.zeros(core_n + core_n * sub_n).reshape(-1,1),
            )
        )
        move_agent_observation_dist_matrix[i,:] = np.where(
            distance.cdist(move_agent_observation_pseudocoordinate,move_agent_observation_pseudocoordinate) == 1.0,1,0
        )
    return (agent_s_s,move_agent_observation_dist_matrix,move_agent_observation)

# generation and visualization of the social networks by caravan-based adjacency matrix
# patitioning was peformed by community.best_partition()
def perform_social_network_analysis(move_agent_observation_dist_matrix,core_n):
    move_agent_observation_dist_matrix_sum = np.sum(move_agent_observation_dist_matrix,axis=0)
    # np.savetxt('move_agent_observation_dist_matrix_sum.csv', move_agent_observation_dist_matrix_sum)
    A = move_agent_observation_dist_matrix_sum
    G = nx.from_numpy_matrix(A,create_using = nx.Graph)
    partition = community.best_partition(G)
    pos = nx.spring_layout(G)
    estimated_core_n = max(partition.values()) + 1
    evaluation = estimated_core_n / core_n
    return (G,partition,evaluation)

# main loops of the simulation
def simulation(agent_s_s,core_n,sub_n,r,experiment_n):
    evaluation_s = []
    for experiment_n in range(1,101):
        agent_s_s,move_agent_observation_dist_matrix,move_agent_observation = perform_observation(core_n,sub_n,experiment_n,r)
        G, evaluation = perform_social_network_analysis(move_agent_observation_dist_matrix,core_n)
        evaluation_s.append(evaluation)
    return evaluation_s

# additional methods for the visualizations of the presentation materials.
# make the figure of the example spatial distributions of the agents.
def mk_fig_initial_spacing(agent_s_s,core_n,sub_n,r,experiment_n,experiment_id = 0):
    plt.close()
    plt.scatter(agent_s_s[:core_n,1,experiment_id],agent_s_s[:core_n,2,experiment_id],s=100,alpha=0.5)
    plt.scatter(agent_s_s[core_n:,1,experiment_id],agent_s_s[core_n:,2,experiment_id],s=100,alpha=0.5)
    for i in np.arange(core_n):
        plt.annotate(i,xy = (agent_s_s[i,1,experiment_id],agent_s_s[i,2,experiment_id]),ha = "center", va = "center",fontsize = 10)
    plt.title('Example group spacing with parameters, \n$n_{I} = %i$, $n_{D} = %i$, $\sigma_{I} = %1.1f$' %(core_n,sub_n,r))
    plt.tight_layout()
    plt.savefig('example_fig/init_%i_%i_%1.1f_%i_%i.png' %(core_n,sub_n,r,experiment_n,experiment_id),dpi=300)

# additional methods for the visualizations of the presentation materials.
# make the figure of the example output of the social network.
def mk_fig_social_network(G,partition,core_n,sub_n,r,experiment_n):
    plt.close()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, node_color=list(partition.values()))
    estimated_core_n = max(partition.values()) + 1
    evaluation = estimated_core_n / core_n
    plt.title('Estimated social networks with parameters, \n$n_{I} = %i$, $n_{D} = %i$, $\sigma_{I} = %1.1f$, $n_{exp} = %i$\nScore = %1.1f' %(core_n,sub_n,r,experiment_n,evaluation))
    plt.tight_layout()
    plt.savefig('example_fig/out_%i_%i_%1.1f_%i.png' %(core_n,sub_n,r,experiment_n),dpi=300)

# additional methods for the visualizations of the presentation materials.
# make the biplot figure of the main result summary representing the comparison of n_{eval} to n_{I}.
def mk_fig_eval_score_by_experiment_n(evaluation_s,core_n,sub_n,r,experiment_n):
    plt.clf()
    plt.plot(range(1,101),evaluation_s)
    plt.title('Estimation of core cluster in core_n = %i, sub_n = %i, ratio = %1.2f' % (core_n,sub_n,r))
    plt.xlabel('experiment numbers')
    plt.ylabel('estimation score (estimated cluster size / truth cluster size)')
    plt.savefig('example_fig/core_%i_sub_%i_ratio_%1.1f_by_experiment_n.png' %(core_n,sub_n,r))

# additional methods for the visualizations of the presentation materials.
# make the figure of the example caravan orders of the agents, corresponding to the initial spatial distribution of the agents.
def mk_fig_caravan(move_agent_observation,core_n,sub_n,r,experiment_n,experiment_id):
    plt.clf()
    fig, ax = plt.subplots(
        figsize = (1,15)
        )

    ax.scatter([2]*30,range(10,30 * 10 + 10,10),s = 500,alpha=0.5)
    for i in range(30):
        ax.annotate(move_agent_observation[experiment_id,i].astype("int"),xy = (2,10*(i+1)),ha = 'center',va='center',fontsize=15)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.savefig('example_fig/init_caravan_%i_%i_%1.1f_%i_%i.png' %(core_n,sub_n,r,experiment_n,experiment_id),dpi=300)
    plt.close()

# additional methods for the visualizations of the presentation materials.
# running to generate the set of the figures, initial spacing, caravans, and social networks.
def mk_fig_example(core_n,sub_n,r,experiment_n):
    os.makedirs('example_fig',exist_ok=True)
    agent_s_s,move_agent_observation_dist_matrix,move_agent_observation = perform_observation(core_n,sub_n,experiment_n,r)
    G, partition, evaluation = perform_social_network_analysis(move_agent_observation_dist_matrix,core_n)
    for i in range(experiment_n):
        mk_fig_initial_spacing(agent_s_s,core_n,sub_n,r,experiment_n,experiment_id=i)
        mk_fig_caravan(move_agent_observation,core_n,sub_n,r,experiment_n,i)
    mk_fig_social_network(G,partition,core_n,sub_n,r,experiment_n)

# main methods for the simulation.
# indicating the parameter settings.
def main():
    core_n_s = np.arange(1,21)
    sub_n_s = np.arange(1,21)
    iteration_n = 100
    r_s = [float(sys.argv[1])]
    experiment_n_s = [int(sys.argv[2])]

    simulation_result_s = np.zeros((iteration_n,len(core_n_s),len(sub_n_s),len(r_s),len(experiment_n_s),6))

    for j,k,l,m in itertools.product(range(len(core_n_s)),range(len(sub_n_s)),range(len(r_s)),range(len(experiment_n_s))):
        core_n = core_n_s[j]
        sub_n = sub_n_s[k]
        r = r_s[l]
        experiment_n = experiment_n_s[m]
        for i in range(iteration_n):
            agent_s_s,move_agent_observation_dist_matrix,move_agent_observation = perform_observation(core_n,sub_n,experiment_n,r)
            G, partition, evaluation = perform_social_network_analysis(move_agent_observation_dist_matrix,core_n)
            simulation_result_s[i,j,k,l,m,:] = [
                evaluation,
                i,
                core_n_s[j],
                sub_n_s[k],
                r_s[l],
                experiment_n_s[m],
            ]
    np.save("results/simulation_results_%1.1f_%i" %(r_s[0],experiment_n_s[0]),simulation_result_s)
    x = core_n_s
    y = sub_n_s
    X, Y = np.meshgrid(x, y)
    Z = np.mean(simulation_result_s[:,:,:,:,:,0],axis=0)[:,:,0,0]
    
    plt.clf()
    cont = plt.contour(X, Y, Z.T, levels = np.arange(0.1,2.1,0.1))
    plt.title("Dependency ratio = %1.1f, \nexperiment number = %i, \niteration number = 100" %(r_s[0],experiment_n_s[0]))
    plt.xlabel("Core number")
    plt.ylabel("Subordinate number")
    plt.xticks(ticks=np.arange(2,21,1))
    plt.yticks(ticks=np.arange(2,21,1))
    cont.clabel(fmt='%1.1f', fontsize=14)
    plt.tight_layout()
    plt.savefig("results/simulation_%1.1f_%i.png" %(r_s[0],experiment_n_s[0]), dpi=300)

if __name__ == "__main__":
    main()