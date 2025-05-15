import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)
# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define dimensions to test
dimensions = [2, 5, 10, 25, 40, 50, 150, 250, 350, 450, 550, 650, 750, 784]
accuracies = []
# Perform PCA and classification for each dimensionality
for dim in dimensions:
    pca = PCA(n_components=dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # NNS classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_pca, y_train)
    # Predict and evaluate
    y_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Dimensionality: {dim}, Accuracy: {acc:.4f}")
# Plot accuracy vs. dimensionality
plt.figure(figsize=(8, 6))
plt.plot(dimensions, accuracies, marker='o', linestyle='-')
plt.xlabel("Dimensionality")
plt.ylabel("Classification Accuracy")
plt.title("Accuracy vs. Dimensionality on MNIST")
plt.grid()
plt.show()
