import numpy as np
import math as m

# Task 1: Basic Vector Operations
# --------------------------------

# Define two vectors
A = np.array([2, 5, 1])
B = np.array([6, 3, 8])

print("A =", A)
print("B =", B)

# 1️⃣ Vector Addition (Element-wise)

# Adds corresponding elements of both vectors.

print("\n1- Vector Addition:")

vector3 = A + B
print(f"{A} + {B} = {vector3}")

# 2️⃣ Vector Subtraction (Element-wise)

# Subtracts corresponding elements of B from A.

print("\n2- Vector Subtraction:")

vector3 = A - B
print(f"{A} - {B} = {vector3}")

# 3️⃣ Vector Multiplication (Element-wise)

# Multiplies each element in A by the corresponding element in B.

print("\n3- Vector Multiplication (Element-wise):")

vector3 = A * B
print(f"{A} x {B} = {vector3}")

# 4️⃣ Scalar Multiplication

# Multiplies each element in B by a constant scalar value.

print("\n4- Scalar Multiplication (c is constant):")

c = 5

vector3 = c + B
print(f"{c} + {B} = {vector3}")

# Compute the magnitudes (norms) of the vectors.

print("\n5- Magnitude of A and B:")

magnitude1 = np.linalg.norm(A)

formatted_output = f"√{magnitude1**2:.0f}"

print(f"||A|| = {formatted_output}")

magnitude2 = np.linalg.norm(B)

formatted_output = f"√{magnitude2**2:.0f}"

print(f"||B|| = {formatted_output}")

print("\n" + "-" * 50 + "\n")

# Task 2: Compute the Dot Product and Angle Between Vectors

# ---------------------------------------------------------

# Compute the dot product of two vectors.

print("\n5- Dot Product:")

vector3 = np.dot(A, B)
print(f"{A} . {B} = {vector3}")

# Compute the cosine similarity.

print("\n7- Cosine Similarity:")



# Compute the angle (in degrees) between the two vectors.

dot_product = np.dot(A, B)
cos_theta = np.dot(A, B) / (magnitude1 * magnitude2)
angle = np.arccos(cos_theta)

formatted_output = f"({dot_product}) / (√({magnitude1**2:.0f}) * √({magnitude2**2:.0f}))"

# Print the formatted result
print(f"cos(θ) = {formatted_output}")


print("\n8- Angle Between Vectors:")

print(f"cos(θ) = {round(angle * (180 / m.pi))}")

print("\n" + "-" * 50 + "\n")

# Task 3: Compute the Projection of One Vector onto Another

# ---------------------------------------------------------

# Compute the projection of A onto B.

print("\n9- Projection of A onto B:")

projection = (np.dot(A, B) / np.dot(B, B)) * B

# Print the formatted result
print(f"Proj_B A = ({dot_product}) / (√({np.dot(B, B)})) × B")


print("\n" + "-" * 50 + "\n")

# Task 4: Compute the Cross Product (For 3D Vectors)

# ---------------------------------------------------------

# Compute the cross product of two 3D vectors.

print("\n10- Cross Product:")

cross_product = np.cross(A, B)
# Print the formatted result
print(f"A × B = det( | i   j   k | \n"
      f"             | {A[0]}  {A[1]}  {A[2]} | \n"
      f"             | {B[0]}  {B[1]}  {B[2]} | )")


print("\n" + "-" * 50 + "\n")

# Task 5: Checking Linear Dependence

# ---------------------------------------------------------

# Check if A and B are linearly dependent.

print("\n11- Checking Linear Dependence:\n")

if np.all(A / B == (A / B)[0]):
    print("Linearly Dependent")
else:
    print("Not Linearly Dependent")


print("\n" + "-" * 50 + "\n")

# Task 6: Normalize a Vector

# ---------------------------------------------------------

# Normalize A to convert it into a unit vector.

print("\n12- Normalized A:")

print("\n" + "-" * 50 + "\n")

# Task 7: Solve a Vector Equation

# ---------------------------------------------------------

# Solve for x and y in the equation xA + yB = C.

# Given vectors A, B, and C, solve the system of equations.

print("\n13- Solution to the Vector Equation:")

print("\n" + "-" * 50 + "\n")

# Task 8: Find the Closest Vector

# ---------------------------------------------------------

# Given a list of vectors, find which vector is closest to a given target vector.

print("\n14- Closest Vector:")

print("\n" + "-" * 50 + "\n")

# Task 9: Compute Vector Similarity in ML

# ---------------------------------------------------------

# Compute cosine similarity between high-dimensional vectors.

print("\n15- Vector Similarity in ML:")
