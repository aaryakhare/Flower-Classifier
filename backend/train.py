from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()

print("Dataset loaded successfully!")

print("\n===== Dataset Information =====")
print("Total Samples:", len(iris.data))
print("\nFeature Names:")
print(iris.feature_names)
print("\nTarget Names:")
print(iris.target_names)
print("\nFirst Flower Measurements:")
print(iris.data[0])
print("\nFirst Flower Species:")
print(iris.target[0])

X = iris.data
y = iris.target

print("\n===== Features and Labels =====")
print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== Training and Testing Data =====")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

knn = KNeighborsClassifier(n_neighbors=3)
print("\nk-NN model created successfully!")

knn.fit(X_train, y_train)
print("k-NN model trained successfully!")

y_pred = knn.predict(X_test)
print("Predictions completed successfully!")

accuracy = knn.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling completed successfully!")

knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)
print("Scaled k-NN model trained successfully!")

y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Predictions using scaled model completed successfully!")

scaled_accuracy = knn_scaled.score(X_test_scaled, y_test)
print(f"\nScaled Model Accuracy: {scaled_accuracy * 100:.2f}%")

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred_scaled, target_names=iris.target_names))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred_scaled))