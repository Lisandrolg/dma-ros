import numpy as np
import matplotlib.pyplot as plt

from graficos import perceptron_plot 


# Ejemplo
entrada = [ [0.7, 1.3], [2.0, 1.1], [1.0, 1.9],
            [3.0, 1.0], [1.5, 2.1]]

#entrada = [[sublista[0] * 1000, sublista[1]] for sublista in entrada]

#entrada = ( entrada - np.mean(entrada, axis=0) ) / np.std(entrada) #promedio de los puntos


salida = [0,0,0,1,1]

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida)

#incializo los graficos
grafico = perceptron_plot(X, Y, 0.01)

# Tamano datos
X_row = X.shape[0]
X_col = X.shape[1]


# Incializo la recta azarosamente
np.random.seed(471787) #mi querida random seed para que las corridas sean reproducibles
W = np.array( np.random.uniform(-0.5, 0.5, size=X_col))
x0 = np.random.uniform(-0.5, 0.5)


# Leer los valores de la última recta encontrada desde un archivo de texto
try:
    rectas = np.genfromtxt('resctas_rosemblatt.txt', delimiter=',')    
    ultima_recta = np.array( rectas[-1]) # La última recta encontrada
    if ultima_recta is not None and any(ultima_recta):
        # Inicializar la recta con los valores obtenidos del archivo
        W = ultima_recta[:-1]  # 
        x0 = ultima_recta[-1]  # 
except:
    print("No se pudieron determinar rectas previas. Se inicia con valores aleatorios.")
    
# Inicializo la iteracion
epoch_limit = 500    # para terminar si no converge
learning_rate = 0.01
modificados = 1      # lo debo poner algo distinto a 0 la primera vez
epoch = 0

while (modificados and (epoch < epoch_limit)):
    epoch += 1
    modificados = 0  #lo seteo en cero

    #recorro siempre TODA la entrada
    for fila in range(X_row):
        # calculo el estimulo suma, producto interno
        estimulo = x0*1 + W[0]*X[fila,0] + W[1]*X[fila,1]

        # funcion de activacion, a lo bruto con un if
        if(estimulo>0):
            y = 1
        else:
            y = 0

        # solo si corresponde actualizo  W y x0
        if(y != Y[fila]):
            modificados += 1  # encontre un registro que esta mal clasificado
            # actualizo W y x0
            W[0] = W[0] + learning_rate * (Y[fila]-y) * X[fila,0]
            W[1] = W[1] + learning_rate * (Y[fila]-y) * X[fila,1]
            x0 =   x0   + learning_rate * (Y[fila]-y) * 1
            print(W)
            grafico.graficar(W, x0, epoch, fila) #grafico

        

grafico.graficar(W, x0, epoch, -1)
print(epoch, W, x0)
try:
    with open('resctas_rosemblatt.txt', 'a') as file:
        # Escribir el string al final del archivo
        file.write(f"{W[0]},{W[1]},{x0}\n")
except:
    print("No se pudo guardar la recta encontrada.")


