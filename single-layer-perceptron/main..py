import numpy as np

#Definir número de epocas
numEpocas = 70000
q = 6

#Definir numero de atributos
peso = np.array([113, 122, 107, 98, 115, 120])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])

#Definir bias
bias = 1

#Entrada do perceptron
X = np.vstack((peso, pH))
Y = np.array([-1, 1, -1, -1, 1, 1])

#Definir taxa de aprendizado
eta = 0.1

#Definir pesos
W = np.zeros([1, 3]) #Duas entradas + bias

# Array para armazenar os erros
e = np.zeros(6)

#Função de ativação
def funcaoAtivacao(u):
  # Função degrau bipolar
    if u >= 0:
        return 1
    else:
        return -1
      
for j in range(numEpocas):
  for k in range(q):
    # Insere o bias no vetor de entrada
    Xb = np.hstack((bias, X[:, k]))
    # Calcular o campo induzido
    V = np.dot(W, Xb) # Equação (5)

    # Calcular a saída do perceptron
    Yr = funcaoAtivacao(V)

    # Caclular o erro: e = (Y - Yr)
    e[k] = Y[k] - Yr

    # Treinamento do perceptron
    W = W + eta*e[k]*Xb

print("Vetor de erro: ", str(e))