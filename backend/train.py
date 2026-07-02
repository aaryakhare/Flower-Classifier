from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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