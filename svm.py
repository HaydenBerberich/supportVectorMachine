import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

# Generate linearly separable data
np.random.seed(0)
X_train_linear = np.r_[np.random.randn(15, 2) - [2, 2], np.random.randn(15, 2) + [2, 2]]
y_train_linear = [0] * 15 + [1] * 15

# Generate non-linearly separable data
X_train_nonlinear = np.r_[np.random.randn(15, 2), np.random.randn(15, 2) + [2, 2]]
y_train_nonlinear = [0] * 15 + [1] * 15

# Generate test data
X_test = np.random.randn(10, 2)
y_test = [0] * 5 + [1] * 5

# Output data points
print("Linearly Separable Training Data Points:")
print(X_train_linear)
print(y_train_linear)

print("\nNon-Linearly Separable Training Data Points:")
print(X_train_nonlinear)
print(y_train_nonlinear)

print("\nTest Data Points:")
print(X_test)
print(y_test)

# Train SVM on linearly separable data
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train_linear, y_train_linear)

# Train SVM on non-linearly separable data
clf_nonlinear = svm.SVC(kernel='linear')
clf_nonlinear.fit(X_train_nonlinear, y_train_nonlinear)

# Predict and calculate errors
y_pred_train_linear = clf_linear.predict(X_train_linear)
y_pred_test_linear = clf_linear.predict(X_test)
train_error_linear = 1 - accuracy_score(y_train_linear, y_pred_train_linear)
test_error_linear = 1 - accuracy_score(y_test, y_pred_test_linear)

y_pred_train_nonlinear = clf_nonlinear.predict(X_train_nonlinear)
y_pred_test_nonlinear = clf_nonlinear.predict(X_test)
train_error_nonlinear = 1 - accuracy_score(y_train_nonlinear, y_pred_train_nonlinear)
test_error_nonlinear = 1 - accuracy_score(y_test, y_pred_test_nonlinear)

# Identify misclassified points
misclassified_train_linear = X_train_linear[y_train_linear != y_pred_train_linear]
misclassified_test_linear = X_test[y_test != y_pred_test_linear]

misclassified_train_nonlinear = X_train_nonlinear[y_train_nonlinear != y_pred_train_nonlinear]
misclassified_test_nonlinear = X_test[y_test != y_pred_test_nonlinear]

# Output results
print("\nLinearly Separable Data:")
print("Support Vectors:", clf_linear.support_vectors_)
print("Train Error:", train_error_linear)
print("Test Error:", test_error_linear)
print("Misclassified Training Points:", misclassified_train_linear)
print("Misclassified Test Points:", misclassified_test_linear)

# Equation of the line for linearly separable data
w_linear = clf_linear.coef_[0]
b_linear = clf_linear.intercept_[0]
print(f"Equation of the line: {w_linear[0]} * x1 + {w_linear[1]} * x2 + {b_linear} = 0")

print("\nNon-Linearly Separable Data:")
print("Support Vectors:", clf_nonlinear.support_vectors_)
print("Train Error:", train_error_nonlinear)
print("Test Error:", test_error_nonlinear)
print("Misclassified Training Points:", misclassified_train_nonlinear)
print("Misclassified Test Points:", misclassified_test_nonlinear)

# Equation of the line for non-linearly separable data
w_nonlinear = clf_nonlinear.coef_[0]
b_nonlinear = clf_nonlinear.intercept_[0]
print(f"Equation of the line: {w_nonlinear[0]} * x1 + {w_nonlinear[1]} * x2 + {b_nonlinear} = 0")

# Plotting
def plot_svm(clf, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.title(title)
    plt.show()

plot_svm(clf_linear, X_train_linear, y_train_linear, "Linearly Separable Data")
plot_svm(clf_nonlinear, X_train_nonlinear, y_train_nonlinear, "Non-Linearly Separable Data")