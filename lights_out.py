import numpy as np

def generar_matriz_A(n):
    size = n*n
    A = np.zeros((size, size), dtype=int)

    #mapeo de posiciones en el vector de salida 
    def idx(i, j):
        return i*n + j

    #generacion de las ecuaciones para cada celda 
    for i in range(n):
        for j in range(n):
            p = idx(i, j)

            #NOTA:toda celda se afecta a si misma 
            A[p][p] = 1  

            #celdas adyacentes
            if i > 0:     A[p][idx(i-1, j)] = 1
            if i < n-1:   A[p][idx(i+1, j)] = 1
            if j > 0:     A[p][idx(i, j-1)] = 1
            if j < n-1:   A[p][idx(i, j+1)] = 1

    return A

#vector b representativo de la matriz inicial
def tablero_a_b(tablero):
    n = len(tablero)
    return np.array(tablero).reshape(n*n) % 2



#aplicacion de gaussiana con suma binaria, sin producto y solo con la operacion Fi -> Fi + Fj
def gauss_mod2(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)

    fila = 0
    for col in range(n):
        #si A[fila][col] == 0 buscar una fila de abajo con 1
        if A[fila][col] == 0:
            for k in range(fila+1, n):
                if A[k][col] == 1:
                    #operacion Fi -> Fi + Fj
                    A[fila] = (A[fila] + A[k]) % 2
                    b[fila] = (b[fila] + b[k]) % 2
                    break
        
        #si sigue siendo 0, no hay pivote en esta columna -> seguir
        if A[fila][col] == 0:
            continue

        #eliminar el resto de la columna (arriba y abajo)
        for k in range(n):
            if k != fila and A[k][col] == 1:
                A[k] = (A[k] + A[fila]) % 2
                b[k] = (b[k] + b[fila]) % 2
        
        fila += 1
        if fila == n:
            break

    return b  #b queda convertido en x



def resolver_lights_out(tablero):
    n = len(tablero)
    A = generar_matriz_A(n)
    b = tablero_a_b(tablero)
    x = gauss_mod2(A, b)
    return x

#tablero de prueba
tablero = [
    [1,0,0],
    [0,0,0],
    [0,1,1]
]

sol = resolver_lights_out(tablero)
print ("Tablero inicial:")
for fila in tablero:
    print(*fila)
print()
print ("El vector solucion es:")
print("["+", ".join(str(num) for num in sol)+"]")
