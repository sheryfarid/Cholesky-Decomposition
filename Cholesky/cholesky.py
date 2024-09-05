
import numpy as np
from scipy.linalg import cholesky

def input_matrix(rows, cols, name):
    print(f"Enter the elements of the matrix {name} row-wise (space-separated):")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        matrix.append(row)
    return np.array(matrix, dtype=float)

def input_vector(size, name):
    print(f"Enter the elements of the vector {name} (space-separated):")
    vector = list(map(float, input(f"{name}: ").split()))
    return np.array(vector, dtype=float)

def cholesky_solve(A, B):
    # Perform Cholesky decomposition
    L = cholesky(A, lower=True)
    
    # Solve LY = B using forward substitution
    Y = np.linalg.solve(L, B)
    
    # Solve L.T X = Y using backward substitution
    X = np.linalg.solve(L.T, Y)
    
    # Set very small values to zero
    Y = np.where(np.abs(Y) < 1e-10, 0, Y)
    X = np.where(np.abs(X) < 1e-10, 0, X)
    
    return L, Y, X

def main():
    rows = int(input("Enter the number of rows and columns for the square matrix A: "))
    cols = rows  # Since A is a square matrix

    # Input matrix A
    A = input_matrix(rows, cols, "A")
    
    # Input vector B
    B = input_vector(rows, "B")

    # Ensure A is symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric.")

    # Ensure A is positive definite by checking its largest eigenvalue
    eigenvalues = np.linalg.eigvals(A)
    largest_eigenvalue = np.max(eigenvalues)
    if largest_eigenvalue <= 0:
        raise ValueError("Matrix A must be positive definite.")
    
    # Print the largest eigenvalue to show it is positive
    print("\nLargest eigenvalue of A (should be positive):")
    print(largest_eigenvalue)

    # Show the calculation of A - λI for the largest eigenvalue
    print("\nCalculation of A - λI (for the largest eigenvalue λ):")
    I = np.identity(rows)
    A_lambda_I = A - largest_eigenvalue * I
    print(f"\nFor λ = {largest_eigenvalue}:")
    print(f"A - λI = A - {largest_eigenvalue} * I")
    print(A_lambda_I)
    
    # Perform the Cholesky method
    L, Y, X = cholesky_solve(A, B)
    
    # Print results
    print("\nMatrix A:")
    print(A)
    
    print("\nVector B:")
    print(B)
    
    print("\nL (lower triangular matrix):")
    print(L)
    
    print("\nL^T (upper triangular matrix):")
    print(L.T)
    
    # Show the calculations for LY = B
    print("\nCalculation of Y (solution of LY = B):")
    for i in range(len(Y)):
        eq = " + ".join([f"{L[i, j]:.2f}*Y{j+1}" for j in range(i+1)]) + f" = {B[i]:.2f}"
        print(eq)
    
    # Print Y solution with exact values
    print("\nY (solution of LY = B):")
    print([f"Y{i+1}:{int(y) if y.is_integer() else f'{y:.2f}'}" for i, y in enumerate(Y)])
    
    # Show the calculations for L^T X = Y
    print("\nCalculation of X (solution of L^T X = Y):")
    for i in range(len(X)-1, -1, -1):
        eq = " + ".join([f"{L.T[i, j]:.2f}*X{j+1}" for j in range(i, len(X))]) + f" = {Y[i]:.2f}"
        print(eq)
    
    # Print X solution with exact values
    print("\nX (solution of L.T X = Y and thus AX = B):")
    print([f"X{i+1}:{int(x) if x.is_integer() else f'{x:.2f}'}" for i, x in enumerate(X)])

print(main())
    