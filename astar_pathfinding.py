import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Create a grid world (0 = free, 1 = obstacle)
grid_size = 20
grid = np.zeros((grid_size, grid_size))

# Add some obstacles (walls)
grid[5:15, 10] = 1  # vertical wall
grid[10, 5:15] = 1  # horizontal wall
grid[3:8, 5] = 1    # small wall
grid[15:18, 15] = 1 # another wall

# Start and goal positions
start = (2, 2)
goal = (18, 18)

def heuristic(pos, goal):
    """Manhattan distance heuristic"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def get_neighbors(pos, grid):
    """Get valid neighboring cells (up, down, left, right)"""
    rows, cols = grid.shape
    r, c = pos
    neighbors = []
    
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    
    return neighbors

def astar(grid, start, goal):
    """A* pathfinding algorithm"""
    # Priority queue: (f_score, counter, position)
    open_set = [(0, 0, start)]
    counter = 0
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    visited = set()
    explored = []  # for visualization
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        explored.append(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], explored
        
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
    
    return None, explored  # No path found

# Run A* algorithm
path, explored = astar(grid, start, goal)

# Visualize
plt.figure(figsize=(12, 10))

# Plot 1: Grid with obstacles
plt.subplot(1, 2, 1)
plt.imshow(grid, cmap='Greys', origin='lower')
plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
if path:
    path_array = np.array(path)
    plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, label='Path')
plt.title('A* Pathfinding Result')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Explored cells
plt.subplot(1, 2, 2)
explored_grid = grid.copy()
for cell in explored:
    if explored_grid[cell] == 0:
        explored_grid[cell] = 0.5
if path:
    for cell in path:
        explored_grid[cell] = 0.7
explored_grid[start] = 0.3
explored_grid[goal] = 0.3
plt.imshow(explored_grid, cmap='YlOrRd', origin='lower')
plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
if path:
    path_array = np.array(path)
    plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, label='Path')
plt.title('Cells Explored by A*')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('astar_pathfinding.png')

if path:
    print(f"Path found! Length: {len(path)} steps")
    print(f"Cells explored: {len(explored)}")
    print(f"Efficiency: {len(path)}/{len(explored)} = {len(path)/len(explored):.2%}")
else:
    print("No path found!")

print("Visualization saved as astar_pathfinding.png")
