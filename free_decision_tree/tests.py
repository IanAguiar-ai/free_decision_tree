if __name__ == "__main__":
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from free_decision_tree import DecisionTree
    import pandas as pd

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
    df = sns.load_dataset("iris")  # ou "tips", "titanic", "penguins", etc.
    df = df.drop(columns = ["species"])
    model = DecisionTree(data = df.iloc[:140], y = "petal_length", max_depth = 5, min_samples = 2)
    print(model)
    model.plot_tree()
    resp1 = model.plot_ci(test = df.iloc[140:])
    model.plot_sensitivity(train = df.iloc[:100], test = df.iloc[100:])
    
    
    def simple_loss_2(y) -> float:
        y_:float = y.mean()
        return 1/len(y) * sum([abs(y_i - y_)*(y_i - y_) for y_i in y])

    model = DecisionTree(data = df, y = "petal_length", max_depth = 5, min_samples = 2, loss_function = simple_loss_2)
    print(model)
    model.plot_tree()
    resp2 = model.plot_ci()
    model.plot_sensitivity(train = df.iloc[:100], test = df.iloc[100:])

    print(f"CI\t{resp1:0.04f}\n\t{resp2:0.04f}")

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

