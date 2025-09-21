if __name__ == "__main__":
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from free_decision_tree import DecisionTree
    import pandas as pd
    import matplotlib.pyplot as plt
    from random import random, seed
    from time import time

##    df = sns.load_dataset("titanic")  # ou "iris", "tips", "titanic", "penguins", etc.
##    df["sex"] = df["sex"].replace({"female": 0, "male": 1})
##    df["alone"] = df["alone"].replace({False: 0, True: 1})
##    df["age"] = df["age"]
##    df = df.drop(columns = list(set(df.columns) ^ {"survived", "age", "sex", "alone"}))#["species"])
##    df = df.dropna()
##    #print(df)
##
##    model = DecisionTree(data = df.iloc[:], y = "survived", max_depth = 4, print = False, train = True)
##    model.plot_ci()
##    #model.plot_sensitivity(train = df.iloc[:int(len(df)*0.7)], test = df.iloc[int(len(df)*0.3):])
##
##    #scaler = StandardScaler()
##    #df_padronizado = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
##
##    #print(df)
##
##    #model = DecisionTree(data = df, y = "petal_length", max_depth = 4, print = False)
####    print(model)
####    print(model.ls)
####    print(model.rs)
####    values = model.predict(df)
####    df["y"] = values
####
####    print(df)
####    print(model(df.iloc[20:20+1]))
####    model.plot_tree()
####

###########################################################################
##    df = sns.load_dataset("iris")  # ou "tips", "titanic", "penguins", etc.
##    df = df.drop(columns = ["species"])
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Carrega o dataset original e remove a coluna 'species'
    df = sns.load_dataset("iris")
    df = df.drop(columns=["species"])

    # Define a seed para reprodutibilidade
    np.random.seed(42)

    # Número de dados sintéticos que você quer gerar
    n_sinteticos = 1000

    # Calcula média e desvio padrão de cada coluna
    means = df.mean()
    stds = df.std()

    # Gera dados sintéticos com distribuição normal baseada nas estatísticas originais
    dados_sinteticos = {
        col: np.random.normal(loc=means[col], scale=stds[col], size=n_sinteticos)
        for col in df.columns
    }

    # Cria um DataFrame com os dados sintéticos
    df_sintetico = pd.DataFrame(dados_sinteticos)

    # Junta os dados originais com os sintéticos
    df = pd.concat([df, df_sintetico], ignore_index=True)

    t0 = time()
    model = DecisionTree(data = df.iloc[:], y = "petal_length", max_depth = 5, min_samples = 2)
    t1 = time()
    print(t1-t0)
    print(model)
    model.plot_tree()
##    resp1 = model.plot_ci(test = df.iloc[140:])
    model.plot_sensitivity(train = df.iloc[:100], test = df.iloc[100:])
##    
##    
##    def simple_loss_2(y) -> float:
##        y_:float = y.mean()
##        return 1/len(y) * sum([abs(y_i - y_)*(y_i - y_) for y_i in y])
##
##    model = DecisionTree(data = df, y = "petal_length", max_depth = 5, min_samples = 2, loss_function = simple_loss_2)
##    print(model)
##    model.plot_tree()
##    resp2 = model.plot_ci()
##    model.plot_sensitivity(train = df.iloc[:100], test = df.iloc[100:])
##
##    print(f"CI\t{resp1:0.04f}\n\t{resp2:0.04f}")

###########################################################################
##    from random import random
##    from time import time
##    
##    n:int = 10_000
##    df = {"a":[random()*100 for i in range(n)],
##          "b":[random()*100 for i in range(n)],
##          "c":[random()*100 for i in range(n)],
##          "d":[random()*100 for i in range(n)]}
##    df:pd.DataFrame = pd.DataFrame(df)
##    print(f"Created Dataframe...")
##
##    t0 = time()
##    model = DecisionTree(data = df.iloc[:1000], y = "d", max_depth = 4, min_samples = 2)
##    t1 = time()
##    print(f"Tempo: {t1-t0}")
##    model.plot_tree()
##    model.plot_ci()
##
##    t0 = time()
##    model.plot_sensitivity(train = df.iloc[:8000], test = df.iloc[8000:])
##    t1 = time()
##    print(f"Tempo: {t1-t0}")
##    
##    def simple_loss_2(y) -> float:
##        y_:float = y.mean()
##        return sum([(y_i - y_)*(y_i - y_) for y_i in y])
##
##    t0 = time()
##    model = DecisionTree(data = df, y = "d", max_depth = 4, min_samples = 2, loss_function = simple_loss_2)
##    t1 = time()
##    print(f"Tempo: {t1-t0}")
##    model.plot_tree()
##    model.plot_ci()

###########################################################################
##    from random import random
##    from math import cos, sin
##    import matplotlib.pyplot as plt
##
##    data = {"x":[cos(i/10)*5 + random() for i in range(100)],
##            "y":[sin(i/10)*5 + random() for i in range(100)],
##            "z":[cos(i/10) + sin(i/10) for i in range(100)]}
##
##    data = pd.DataFrame(data)
##    model = DecisionTree(data = data, y = "z", max_depth = 7)
##    model.plot_tree()
##    nd = model.detect_depth()
##    print(nd)
##
##    plt.figure(figsize = (10, 10))
##    plt.scatter(nd["x"], nd["y"], c = nd["__dt_depth__"], alpha = 0.8)
##    plt.show()

###########################################################################
##    df = sns.load_dataset("iris")  # ou "tips", "titanic", "penguins", etc.
##    df = df.drop(columns = ["species", "petal_width"])
##    model = DecisionTree(data = df.iloc[:140], y = "petal_length", max_depth = 3, min_samples = 2)
##    #print(model)
##    y = model.predict_smooth(df)
##    print(set(sorted(y)))
##    #model.plot_tree()
##    plt.figure(figsize = (7, 7))
##    plt.scatter(df["sepal_length"], df["sepal_width"], c = y, alpha = 0.4)
##    plt.show()

###########################################################################
##    seed(1)
##    df = pd.DataFrame({"a":[*[0, 1, 0, 1], *[1 + random()-0.5 for i in range(10)]],
##                       "b":[*[0, 0, 1, 1], *[1 + random()-0.5 for i in range(10)]],
##                       "c":[*[0, 1, -1, 2], *[2 + random()/10 for _ in range(10)]]})
##    model = DecisionTree(data = df, y = "c", max_depth = 2, min_samples = 1)
##    
##    df_test = pd.DataFrame({"a":[(i/41)%1 for i in range(1000)], "b":[(i/67)%1 for i in range(1000)]})
##    y = model.predict_smooth(df_test, alpha = .001)
##    plt.figure(figsize = (6, 4))
##    plt.scatter(df_test["a"], df_test["b"], c = y, alpha = 1)
##    plt.show()
##    model.plot_tree()
##
##    df_ = model.detect_depth()
##    print(df_)

###########################################################################
##df = sns.load_dataset("flights")  # ou "tips", "titanic", "penguins", etc.
##meses:list = [mes for mes in df["month"].iloc[:12]]
##df["month"] = df["month"].replace({mes:int(i+1) for i, mes in enumerate(meses)})
##df["month"] = df["month"].astype(int)
##
##print(df)
##
##def simple_loss_3(y:pd.DataFrame) -> float:
##    y_:float = y.mean()
##    return sum([(y_i - y_)**8 for y_i in y])
##
##model3 = DecisionTree(data = df.iloc[::2], y = "passengers", max_depth = 5, min_samples = 1,
##                      loss_function = simple_loss_3, loss_calc = lambda a, b : max(a, b))
##
##df_temp = df[df["year"] >= 1949]
##X = [df_temp["month"].iloc[i]/12 + df_temp["year"].iloc[i] for i in range(len(df_temp))]
##
##plt.figure(figsize = (14, 7))
##plt.plot(X, model3(df_temp), color = "orange", label = "Predito (Modelo 3)", alpha = 0.7)
##plt.plot(X, model3.predict_smooth(df_temp, representatives = False, n_neighbors = 3), color = "purple", linestyle = "--", label = "Predito (Modelo 3 com técnica de smooth)", alpha = 0.7)
##plt.plot(X, model3.predict_smooth(df_temp, representatives = True, n_neighbors = 3), color = "pink", linestyle = ":", label = "Predito (Modelo 3 com técnica de smooth e representative)", alpha = 0.7)
##
##
##plt.plot(X, df_temp["passengers"], color = "red", linestyle = "--", label = "Real")
##
##plt.xlabel("Ano")
##plt.ylabel("Passageiros")
##plt.grid()
##plt.legend()
##plt.show()

###########################################################################
##df = sns.load_dataset("flights")  # ou "tips", "titanic", "penguins", etc.
##meses:list = [mes for mes in df["month"].iloc[:12]]
##df["month"] = df["month"].replace({mes:int(i+1) for i, mes in enumerate(meses)})
##df["month"] = df["month"].astype(int)
##
##print(df)
##
##def simple_loss_3(y:pd.DataFrame) -> float:
##    y_:float = y.mean()
##    return sum([(y_i - y_)**8 for y_i in y])
##
##model3 = DecisionTree(data = df.iloc[::2], y = "passengers", max_depth = 5, min_samples = 1,
##                      loss_function = simple_loss_3)
##
##
##model3.plot_tree()
##
##model3.save("test")
##
##new_model = DecisionTree.load("test")
##new_model.plot_tree()
