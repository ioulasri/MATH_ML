import numpy as np

# Vector Operations

vector1 = np.array([1, 2, 3])
vector2 = np.array([3, 4, 5])

print("Vector1 = ", vector1)
print("Vector2 = ", vector2)

# Vector Addition (Element-wise)

vector3 = vector1 + vector2

print("\nVector Addition:")
print(f"{vector1} + {vector2} = {vector3}")

# Vector Multiplication (Element-wise)

vector3 = vector1 * vector2

print("\nVector Multiplication (Element-wise):")
print(f"{vector1} x {vector2} = {vector3}")

# Dot Product of Vectors

dot_product = np.dot(vector1, vector2)

print("\nVector Dot Product:")
print(f"{vector1} · {vector2} = {dot_product}")

print("\n---\n")

# Matrix Operations

matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[-1, -2, -3], [7, 8, 9]])

print("Matrix1 =\n", matrix1)
print("Matrix2 =\n", matrix2)

# Matrix Addition (Element-wise)

matrix3 = matrix1 + matrix2

print("\nMatrix Addition:")
print(f"{matrix1}\n + \n{matrix2}\n =\n{matrix3}")

# Matrix Multiplication

matrix3 = np.dot(matrix1, matrix2.T)  # or matrix1 @ matrix2.T

print("\nMatrix Multiplication (Using .T to transpose the matrix for shape compatibility):")
print(f"{matrix1}\n x \n{matrix2.T}\n =\n{matrix3}")

print("\n---\n")

# Matrix Transpose

print("\nMatrix Transpose:")
print(f"Transpose of Matrix1:\n{matrix1.T}")

# Matrix Determinant (only works for square matrices)

try:
    determinant = np.linalg.det(matrix1 @ matrix1.T)
    print("\nDeterminant of (Matrix1 * Matrix1.T):", round(determinant, 2))
except np.linalg.LinAlgError:
    print("\nDeterminant cannot be computed for this matrix.")
