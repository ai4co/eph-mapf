import numpy as np
import matplotlib.pyplot as plt


def plot_mapf(map_, init_pos, goal_pos):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Display the map
    ax.imshow(map_, cmap='binary', origin='lower')

    # # Define colors for each agent to distinguish them
    # colors = ['red', 'blue', 'green', 'orange']
    num_agents = init_pos.shape[0]
    colors = plt.cm.prism(np.linspace(0, 1, num_agents))

    # Plot grid lines every 1 unit
    ax.hlines(y=np.arange(.5, map_.shape[0]-1.5), xmin=0, xmax=map_.shape[1]-.5, color='k', linewidth=0.5, alpha=0.5)
    ax.vlines(x=np.arange(.5, map_.shape[1]-1.5), ymin=0, ymax=map_.shape[0]-.5, color='k', linewidth=0.5, alpha=0.5)

    # Plot initial and goal positions
    for i in range(num_agents):
        ax.plot(init_pos[i, 1], init_pos[i, 0], 'o', color=colors[i], label=f'Agent {i+1} Start')
        ax.plot(goal_pos[i, 1], goal_pos[i, 0], 'x', color=colors[i], label=f'Agent {i+1} Goal')

    # remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # remove unoplotted margins
    plt.margins(0)


    plt.tight_layout()
    plt.show()