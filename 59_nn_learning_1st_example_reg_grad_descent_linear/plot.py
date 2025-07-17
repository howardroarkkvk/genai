import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define loss function: z = x^2 + y^2 and its gradient
def grad(x, y):
    return 2 * x, 2 * y

# Create a surface
x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2

# Hyperparameters
lr = 0.2
batches_per_epoch = 5
epochs = 3
total_steps = batches_per_epoch * epochs

# Starting point
x, y = 3.5, 3.5
positions = [(x, y)]

# Simulate gradient descent steps
for _ in range(total_steps):
    dx, dy = grad(x, y)
    x -= lr * dx
    y -= lr * dy
    positions.append((x, y))

# Extract values for plotting
x_path, y_path = zip(*positions)
z_path = [x**2 + y**2 for x, y in positions]

# Plotting the 3D surface and the GD steps
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.plot(x_path, y_path, z_path, color='red', marker='o', label='GD Step (per batch)')

# Highlight points at the end of each epoch
for i in range(1, epochs):
    idx = i * batches_per_epoch
    ax.scatter(x_path[idx], y_path[idx], z_path[idx], color='blue', s=60, label=f'End of Epoch {i}')

ax.set_title("Gradient Descent Steps Across Batches and Epochs")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Loss")
ax.legend()
plt.tight_layout()
plt.show()
