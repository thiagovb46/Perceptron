from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
atributos = iris.data
classes = iris.target

atributos_do_conjunto = atributos[classes != 0]
classes_conjunto = classes[classes != 0]

classes_conjunto = np.where(classes_conjunto == 1, 1, -1)

taxa_aprendizado = 0.1;

for cont in range(3):
    if cont == 0:
        taxa_aprendizado = 0.1
        proporcao_conn_teste = 0.1
    else:
        if cont == 1:
            taxa_aprendizado = 0.2
            proporcao_conn_teste = 0.3
        else: 
            if cont == 2:
                taxa_aprendizado = 0.3
                proporcao_conn_teste = 0.5
    # Dividir o conjunto de dados em treinamento e teste
    atributos_treinamento, atributos_teste, classes_treinamento, classes_teste = train_test_split(atributos_do_conjunto, classes_conjunto, test_size=proporcao_conn_teste, random_state=42)
    tamanho_conjunto = atributos_teste.shape[0]
    #Inicialização pesos/bias
    np.random.seed(0)
    pesos = np.random.rand(atributos_treinamento.shape[1])
    viés = np.random.rand()

    # Treino com 100 interações
    maximo_interacaoes = 100
    for epoca in range(maximo_interacaoes):
        for i in range(atributos_treinamento.shape[0]): #Tamanho conjunto treinamento
            entrada = atributos_treinamento[i]
            objetivo = classes_treinamento[i]
            soma_ponderada = np.dot(entrada, pesos) + viés #Soma ponderada das entradas do neuronio + vies
            ativação = np.sign(soma_ponderada)
            if ativação != objetivo:
                erro = objetivo - ativação
                pesos += taxa_aprendizado * erro * entrada
                viés += taxa_aprendizado * erro

    # Testes
    corretos = 0
    for i in range(tamanho_conjunto): #Tamanho conjunto de testes
        entrada = atributos_teste[i]
        objetivo = classes_teste[i]
        soma_ponderada = np.dot(entrada, pesos) + viés
        ativação = np.sign(soma_ponderada)
        if ativação == objetivo:
            corretos += 1

    acuracia = (corretos / tamanho_conjunto) * 100 #Quantidade de acertos/tamanho do conjunto
    print("Conjunto "+str(cont+1))
    print("     Taxa de aprendizado:", taxa_aprendizado)
    print("     Tamanho conjunto de teste:", proporcao_conn_teste)
    print("     Acurácia no conjunto de teste %:", acuracia)
    print("---------------------------------------")