import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# Data points
pts = np.array([
    [8, 5], [13, 1], [13, 9],  # Class +1
    [1, 9], [5, 5], [1, 1]     # Class -1
])

lbls = [1, 1, 1, -1, -1, -1]  # Class labels

# Fit SVM with linear kernel
clf = svm.SVC(kernel='linear')
clf.fit(pts, lbls)

# Plot points
fig = plt.figure(figsize=(8, 6))
colors = ['red' if label == 1 else 'blue' for label in lbls]
markers = ['o' for label in lbls]

for i in range(len(pts)):
    plt.scatter(pts[i][0], pts[i][1], color=colors[i], marker=markers[i], s=100)

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
            linestyles=['--', '-', '--'])

# Support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
            s=150, linewidth=1, facecolors='none', edgecolors='k')

# Margin calculation
margin = 2/np.linalg.norm(clf.coef_)
print("Margin:", margin)

plt.xticks(np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 1, 1))
plt.yticks(np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) + 1, 1))

fig.canvas.manager.set_window_title("SVM Hyperplane and Maximum Margin")
plt.title("SVM Hyperplane and Maximum Margin")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
