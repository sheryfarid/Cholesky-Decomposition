from flask import Flask, request, render_template
import numpy as np
from scipy.linalg import cholesky

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            rows = int(request.form['rows'])
            matrix_input = request.form['matrix'].strip().split('\n')
            vector_input = request.form['vector'].strip().split()

            A = np.array([list(map(float, row.split())) for row in matrix_input])
            B = np.array(list(map(float, vector_input)))

            if not np.allclose(A, A.T):
                raise ValueError("Matrix A must be symmetric.")

            eigenvalues = np.linalg.eigvals(A)
            largest_eigenvalue = np.max(eigenvalues)
            if largest_eigenvalue <= 0:
                raise ValueError("Matrix A must be positive definite.")
            
            L, Y, X = cholesky_solve(A, B)

            result = {
                "largest_eigenvalue": largest_eigenvalue,
                "A_lambda_I": A - largest_eigenvalue * np.identity(rows),
                "A": A,
                "B": B,
                "L": L,
                "LT": L.T,
                "Y": Y,
                "X": X
            }

            return render_template('index.html', result=result)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
