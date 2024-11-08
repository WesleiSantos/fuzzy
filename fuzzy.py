import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Função para ler os dados
def read_data(filename):
    class_mapping = {'Iris-setosa': [1, 0], 'Iris-versicolor': [0, 1], 'Iris-virginica': [0, 0]}  # Mapeia as classes para binário
    
    X = []
    d = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@') or not line.strip():  # Ignora as linhas de metadados e vazias
                continue
            data = line.strip().split(',')
            X.append([float(x) for x in data[:-1]])  # Características
            d.append(data[-1].strip())  # Classe convertida para valor binário
    return np.array(X), np.array(d)


def plot_histograms(X, d):
    # Criar DataFrame
    df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['class'] = d

    # Cores para cada classe
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

    # Criar subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Facilita a iteração

    # Nomes das colunas
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Plotar histogramas
    for i, column in enumerate(columns):
        for class_name, color in colors.items():
            subset = df[df['class'] == class_name]
            axs[i].hist(subset[column], bins=20, alpha=0.5, color=color, label=class_name)

        axs[i].set_title(f'Histograma de {column.replace("_", " ").title()}')
        axs[i].set_xlabel(column.replace("_", " ").title())
        axs[i].set_ylabel('Frequência')
        axs[i].legend(title='Classes')

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.show()


X, d= read_data('iris.dat')
#plot_histograms(X, d)


# Função de pertinência triangular
def triangular(x, a, m, b):
    return np.maximum(0, np.minimum((x - a) / (m - a), (b - x) / (b - m)))

# Intervalos das variáveis para os gráficos
x_sepal_length = np.linspace(4.0, 8.0, 200)
x_sepal_width = np.linspace(2.0, 4.5, 200)
x_petal_length = np.linspace(1.0, 7.0, 200)
x_petal_width = np.linspace(0.1, 2.5, 200)

# Definição das funções de pertinência para cada conjunto fuzzy
fuzzy_sets = {
    "sepal_length": {
        "small": lambda x: triangular(x, 4.0, 5, 5.3),
        "medium": lambda x: triangular(x, 5.3, 5.5, 6.5),
        "large": lambda x: triangular(x, 6.5, 6.8, 8.0)
    },
    "sepal_width": {
        "small": lambda x: triangular(x, 2.0, 2.4, 2.9),
        "medium": lambda x: triangular(x, 2.9, 3.2, 3.3),
        "large": lambda x: triangular(x, 3.3, 3.5, 4.5)
    },
    "petal_length": {
        "small": lambda x: triangular(x, 1.0, 1.5, 3.0),
        "medium": lambda x: triangular(x, 3.0, 4.0, 5.0),
        "large": lambda x: triangular(x, 5.0, 5.5, 7.0)
    },
    "petal_width": {
        "small": lambda x: triangular(x, 0.1, 0.3, 0.6),
        "medium": lambda x: triangular(x, 0.6, 1.0, 1.5),
        "large": lambda x: triangular(x, 1.5, 2.0, 2.5)
    }
}

# Definição das regras fuzzy
rules = [
    {"sepal_length": "small", "sepal_width": "large", "petal_length": "small", "petal_width": "small", "class": "Iris-setosa"},
    {"sepal_length": "small", "sepal_width": "medium", "petal_length": "small", "petal_width": "small", "class": "Iris-setosa"},
    {"sepal_length": "medium", "sepal_width": "large", "petal_length": "small", "petal_width": "small", "class": "Iris-setosa"},
    {"sepal_length": "medium", "sepal_width": "medium", "petal_length": "small", "petal_width": "small", "class": "Iris-setosa"},
    {"sepal_length": "medium", "sepal_width": "small", "petal_length": "medium", "petal_width": "medium", "class": "Iris-versicolor"},
    {"sepal_length": "medium", "sepal_width": "medium", "petal_length": "medium", "petal_width": "medium", "class": "Iris-versicolor"},
    {"sepal_length": "small", "sepal_width": "small", "petal_length": "medium", "petal_width": "medium", "class": "Iris-versicolor"},
    {"sepal_length": "small", "sepal_width": "medium", "petal_length": "medium", "petal_width": "medium", "class": "Iris-versicolor"},
    {"sepal_length": "large", "sepal_width": "medium", "petal_length": "large", "petal_width": "large", "class": "Iris-virginica"},
    {"sepal_length": "medium", "sepal_width": "medium", "petal_length": "large", "petal_width": "large", "class": "Iris-virginica"},
    {"sepal_length": "large", "sepal_width": "small", "petal_length": "large", "petal_width": "large", "class": "Iris-virginica"},
    {"sepal_length": "medium", "sepal_width": "large", "petal_length": "large", "petal_width": "medium", "class": "Iris-virginica"},
]

# Função para plotar os conjuntos fuzzy
def plot_fuzzy_sets(x_sepal_length, x_sepal_width, x_petal_length, x_petal_width, fuzzy_sets):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot do sepal_length
    for label in fuzzy_sets["sepal_length"].keys():
        axs[0, 0].plot(x_sepal_length, fuzzy_sets["sepal_length"][label](x_sepal_length), label=label)
    axs[0, 0].set_title("Sepal Length")
    axs[0, 0].set_xlabel("Sepal Length (cm)")
    axs[0, 0].set_ylabel("Membership Degree")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot do sepal_width
    for label in fuzzy_sets["sepal_width"].keys():
        axs[0, 1].plot(x_sepal_width, fuzzy_sets["sepal_width"][label](x_sepal_width), label=label)
    axs[0, 1].set_title("Sepal Width")
    axs[0, 1].set_xlabel("Sepal Width (cm)")
    axs[0, 1].set_ylabel("Membership Degree")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot do petal_length
    for label in fuzzy_sets["petal_length"].keys():
        axs[1, 0].plot(x_petal_length, fuzzy_sets["petal_length"][label](x_petal_length), label=label)
    axs[1, 0].set_title("Petal Length")
    axs[1, 0].set_xlabel("Petal Length (cm)")
    axs[1, 0].set_ylabel("Membership Degree")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot do petal_width
    for label in fuzzy_sets["petal_width"].keys():
        axs[1, 1].plot(x_petal_width, fuzzy_sets["petal_width"][label](x_petal_width), label=label)
    axs[1, 1].set_title("Petal Width")
    axs[1, 1].set_xlabel("Petal Width (cm)")
    axs[1, 1].set_ylabel("Membership Degree")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# Método para classificar um exemplo com base nas regras fuzzy usando T-norma min
def classify(example):
    max_compatibility = 0
    predicted_class = None

    for rule in rules:
        # Calcula o grau de compatibilidade da regra atual usando a T-norma máxima
        compatibility = min(
            fuzzy_sets["sepal_length"][rule["sepal_length"]](example[0]),
            fuzzy_sets["sepal_width"][rule["sepal_width"]](example[1]),
            fuzzy_sets["petal_length"][rule["petal_length"]](example[2]),
            fuzzy_sets["petal_width"][rule["petal_width"]](example[3])
        )

        # Atualiza a classe predita se o grau de compatibilidade for maior
        if compatibility > max_compatibility:
            max_compatibility = compatibility
            predicted_class = rule["class"]

    return predicted_class


# Método para classificar um exemplo com base no MRFG usando T-norma min e operador de agregação f
def classify_mrfg(example, fuzzy_sets, rules, aggregation_operator=np.max):
    # Inicializar um dicionário para armazenar os graus de compatibilidade para cada classe
    class_compatibility = {rule["class"]: 0 for rule in rules}

    # Para cada regra, calcula o grau de compatibilidade usando T-norma min
    for rule in rules:
        compatibility = min(
            fuzzy_sets["sepal_length"][rule["sepal_length"]](example[0]),
            fuzzy_sets["sepal_width"][rule["sepal_width"]](example[1]),
            fuzzy_sets["petal_length"][rule["petal_length"]](example[2]),
            fuzzy_sets["petal_width"][rule["petal_width"]](example[3])
        )

        # Acumula o grau de compatibilidade para a classe da regra
        class_compatibility[rule["class"]] += compatibility

    # Calcula a média aritmética para cada classe
    class_mean_compatibility = {cls: np.mean(compatibilities) for cls, compatibilities in class_compatibility.items()}

    # Aplica o operador MAX para selecionar a classe com maior média de compatibilidade
    predicted_class = max(class_mean_compatibility, key=class_mean_compatibility.get)
    
    return predicted_class

# Função para avaliar a acurácia com o MRFG
def evaluate_mrfg(X, d, fuzzy_sets, rules):
    correct_predictions = 0

    for i in range(len(X)):
        predicted_class = classify_mrfg(X[i], fuzzy_sets, rules)
        if predicted_class == d[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / len(X)) * 100
    return accuracy

# Função para avaliar a acurácia
def evaluate(X, d):
    correct_predictions = 0

    for i in range(len(X)):
        predicted_class = classify(X[i])
        if predicted_class == d[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / len(X)) * 100
    return accuracy

# Leitura dos dados e avaliação
X, d = read_data('iris.dat')
# Chamando a função para plotar os gráficos
plot_fuzzy_sets(x_sepal_length, x_sepal_width, x_petal_length, x_petal_width, fuzzy_sets)
accuracy = evaluate(X, d)
print(f'Acurácia: {accuracy:.2f}%')

# Calcular a acurácia usando o MRFG
accuracy_mrfg = evaluate_mrfg(X, d, fuzzy_sets, rules)
print(f'Acurácia do Método de Raciocínio Fuzzy Geral (MRFG): {accuracy_mrfg:.2f}%')