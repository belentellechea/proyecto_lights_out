import numpy as np
from itertools import product

def lights_out_solver(initial_state):
    n = initial_state.shape[0]
    size = n * n

    # Construir matriz A
    A = np.zeros((size, size), dtype=int)
    def idx(i, j): return i * n + j

    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            A[k, k] = 1
            if i > 0:     A[k, idx(i - 1, j)] = 1
            if i < n - 1: A[k, idx(i + 1, j)] = 1
            if j > 0:     A[k, idx(i, j - 1)] = 1
            if j < n - 1: A[k, idx(i, j + 1)] = 1

    b = initial_state.flatten() % 2

    # Reducción por filas (mod 2)
    A_ext = np.concatenate((A, b.reshape(-1, 1)), axis=1)
    A_ext = A_ext % 2
    m, ncols = A_ext.shape
    row = 0
    pivots = []
    for col in range(size):
        pivot_rows = np.where(A_ext[row:, col] == 1)[0]
        if pivot_rows.size == 0:
            continue
        pivot = pivot_rows[0] + row
        A_ext[[row, pivot]] = A_ext[[pivot, row]]
        pivots.append(col)
        for r in range(row + 1, m):
            if A_ext[r, col] == 1:
                A_ext[r] = (A_ext[r] + A_ext[row]) % 2
        row += 1
        if row == m: break

    # Sustitución hacia atrás
    x = np.zeros(size, dtype=int)
    for r in reversed(range(row)):
        pivot_col = np.where(A_ext[r, :size] == 1)[0]
        if len(pivot_col) == 0:
            continue
        c = pivot_col[0]
        rhs = (A_ext[r, -1] + np.dot(A_ext[r, c+1:size], x[c+1:])) % 2
        x[c] = rhs

    # Comprobación
    result = (A @ x) % 2
    if np.array_equal(result, b):
        return x.reshape((n, n))
    else:
        # Buscar otra combinación en el espacio nulo
        from scipy.linalg import null_space
        nulls = null_space(A % 2)
        if nulls.size == 0:
            print("No existe solución exacta")
            return None
        nulls = (nulls % 2 > 0.5).astype(int)
        for combo in product([0, 1], repeat=nulls.shape[1]):
            x_test = (x + np.dot(nulls, combo)) % 2
            if np.array_equal((A @ x_test) % 2, b):
                return x_test.reshape((n, n))
        print("Ninguna combinación apaga todas las luces")
        return None

initial = np.array([
    [0, 1, 1],
    [0, 1, 1],
    [1, 0, 0]
])

initial = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

initial = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1]
])

solution = lights_out_solver(initial)
print("Solución encontrada:")
print(solution)


