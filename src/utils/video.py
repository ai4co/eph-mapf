from multiprocessing import Pool
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from functools import partial

# Define the function to save each frame as an image
def save_frame(frame_index, map_, init_pos, goal_pos, positions, frames_dir='/tmp/agent_frames/'):

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

        # ax.plot(init_pos[i, 1], init_pos[i, 0], 'o', color=colors[i], label=f'Agent {i+1} Start')
        ax.plot(goal_pos[i, 1], goal_pos[i, 0], 'x', color=colors[i], label=f'Agent {i+1} Goal')
        ax.plot(positions[frame_index, i, 1], positions[frame_index, i, 0], 'o', color=colors[i], markersize=5, label=f'Agent {i+1} Position' if frame_index == 0 else "")


    # remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # remove unoplotted margins
    plt.margins(0)


    plt.tight_layout()
    plt.show()
    ax.axis('off')
    
    # Save the frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_index:04d}.png')
    fig.savefig(frame_path)
    plt.close(fig)  # Close the plot to free memory
    

def create_video(map_, 
                 init_pos, 
                 goal_pos, 
                 positions,
                frames_dir = '/tmp/agent_frames/',
                filename = './videos/agents_movement.mp4'):

 

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    if not os.path.exists('./videos'):
        os.makedirs('./videos')

    # remove previous frames
    for file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, file))

    # Number of processes to use
    num_processes = 10

    # # Create a partial function with fixed arguments
    # save_frame_partial = partial(save_frame, map_=map_, init_pos=init_pos, goal_pos=goal_pos, positions=positions, frames_dir=frames_dir)

    # with Pool(processes=num_processes) as pool:
    #     results = list(tqdm(pool.imap(save_frame_partial, range(positions.shape[0])), total=positions.shape[0]))
    # Prepare the arguments for each call to save_frame
    args_list = [(i, map_, init_pos, goal_pos, positions, frames_dir) for i in range(positions.shape[0])]

    # Use a multiprocessing pool to parallelize frame saving
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(save_frame, args_list), total=len(args_list)))

    # Construct the ffmpeg command to create a video from the frames
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '10',  # Set frame rate
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),  # Input frames format
        '-c:v', 'libx264',  # Codec to use
        '-pix_fmt', 'yuv420p',  # Pixel format
        '-crf', '23',  # Constant Rate Factor for quality
        '-y', # overwrite is true
        filename  # Output file path
    ]

    # Execute the ffmpeg command
    subprocess.run(ffmpeg_cmd, check=True)