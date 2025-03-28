import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

# Define the nonconvex function
def f(a1, a2):
    term1 = 2 * np.exp(-((a1 - 1)**2 + (1.4*a2)**2)/0.2)
    term2 = 6 * np.exp(-((a1 + 1)**2 + (1.4*a2)**2)/0.2)
    term3 = -3 * np.exp(-((a1 - 1)**2 + (1.4*a2 + 1)**2)/0.2)
    term4 = -0.2 * np.exp(-((a1)**2 + (1.4*a2 - 1)**2)/0.2)
    term5 = a1**2 + (1.4*a2)**2
    return term1 + term2 + term3 + term4 + term5

# Create a meshgrid
a1_range = np.linspace(-2.5, 2.5, 100)
a2_range = np.linspace(-2.5, 2.5, 100)
a1_grid, a2_grid = np.meshgrid(a1_range, a2_range)
z = f(a1_grid, a2_grid)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Add lighting for metallic effect
ax.set_facecolor('black')  # Black background enhances metallic look
fig.patch.set_facecolor('black')

# Set lighting and material properties for metallic look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.grid(True, color='white', alpha=0.3, linestyle='-', linewidth=0.3)

# Normalize z for colormapping
norm = plt.Normalize(z.min(), z.max())
metallic_cmap = plt.cm.coolwarm

# Function to update lighting based on current view
def update_lighting(event=None):
    if hasattr(update_lighting, 'surf') and update_lighting.surf is not None:
        update_lighting.surf.remove()
    
    # Get current view angles
    elev, azim = ax.elev, ax.azim
    
    # Adjust light source to be slightly offset from viewing angle
    # This creates the illusion that light is coming from the viewer's direction
    light_azim = (azim + 30) % 360
    light_elev = max(elev - 15, 5)  # Keep light a bit above view angle
    
    # Create a light source based on current view
    ls = LightSource(azdeg=light_azim, altdeg=light_elev)
    
    # Apply the light source to the surface with a metallic colormap
    rgb = plt.cm.ScalarMappable(cmap=metallic_cmap).to_rgba(norm(z))
    illuminated_surface = ls.shade_rgb(rgb[:,:,:3], z)
    
    # Plot the surface with reflective appearance
    update_lighting.surf = ax.plot_surface(a1_grid, a2_grid, z, 
                          facecolors=illuminated_surface,
                          linewidth=0, 
                          antialiased=True,
                          alpha=0.9)  # Slightly reduced alpha to make points more visible
    
    # Force redraw
    fig.canvas.draw_idle()

# Initialize the surface
update_lighting.surf = None
update_lighting()  # Initial draw

# Define critical points
# Global minimum at (1, -1/1.4)
global_min = (1, -1/1.4)
global_min_z = f(global_min[0], global_min[1])

# Local minimum at (0, 1/1.4)
local_min = (0, 1/1.4)
local_min_z = f(local_min[0], local_min[1])

# Local maxima at (1, 0) and (-1, 0)
local_max1 = (1, 0)
local_max1_z = f(local_max1[0], local_max1[1])
local_max2 = (-1, 0)
local_max2_z = f(local_max2[0], local_max2[1])

# Calculate z range
z_min = z.min() - 0.5  # Slightly below the surface minimum
z_max = z.max() + 1

# Function to create vertical stems for the points with floating text
def add_point_with_stem(x, y, z, color, label=None, size=200):
    # Plot the stem
    ax.plot([x, x], [y, y], [z_min, z], color=color, linestyle='-', linewidth=2, alpha=0.7)
    
    # Plot the point with black edge for better visibility
    scatter = ax.scatter(x, y, z, color=color, s=size, edgecolor='black', linewidth=1.5, label=label)
    
    # Add text label significantly above the point
    x_offset = 0.1  # Small horizontal offset
    y_offset = 0.1
    z_height = 3.0  # Height above the point for text
    
    # Draw a thin vertical line from point to text
    ax.plot([x, x+x_offset], [y, y+y_offset], [z, z+z_height], color=color, linestyle='--', linewidth=1, alpha=0.7)
    
    # Add the text above
    ax.text(x+x_offset, y+y_offset, z+z_height, label, color=color, fontsize=12, fontweight='bold',
           ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.4, edgecolor=color, pad=2))
    
    return scatter

# Add points with stems, larger size, and floating labels
add_point_with_stem(global_min[0], global_min[1], global_min_z, 'lime', 'Global Min', 250)
add_point_with_stem(local_min[0], local_min[1], local_min_z, 'cyan', 'Local Min', 250)
add_point_with_stem(local_max1[0], local_max1[1], local_max1_z, 'orange', 'Local Max 1', 250)
add_point_with_stem(local_max2[0], local_max2[1], local_max2_z, 'magenta', 'Local Max 2', 250)

# Add labels and title
ax.set_xlabel('a1', color='white', fontsize=12)
ax.set_ylabel('a2', color='white', fontsize=12)
ax.set_zlabel('f(a1, a2)', color='white', fontsize=12)
ax.set_title('Non-convex Function', color='white', fontsize=14)

# Set tick colors to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Connect to view change event - this is the key for dynamic lighting
fig.canvas.mpl_connect('motion_notify_event', lambda event: update_lighting() 
                      if event.inaxes == ax and hasattr(event, 'button') and event.button is not None else None)

# Set initial view for better visibility of all points
ax.view_init(elev=40, azim=30)

# Set axis limits to ensure everything is visible including floating labels
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(z_min, z_max + 5)  # Extra space for floating labels

plt.tight_layout()
plt.show()