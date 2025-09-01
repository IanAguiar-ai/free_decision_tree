if __name__ == "__main__":
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from decision_tree.free_decision_tree import DecisionTree

    df = sns.load_dataset("titanic")  # ou "iris", "tips", "titanic", "penguins", etc.
    df["sex"] = df["sex"].replace({"female": 0, "male": 1})
    df["alone"] = df["alone"].replace({False: 0, True: 1})
    df["age"] = df["age"]
    df = df.drop(columns = list(set(df.columns) ^ {"survived", "age", "sex", "alone"}))#["species"])
    df = df.dropna()
    #print(df)

    model = DecisionTree(data = df.iloc[:], y = "survived", max_depth = 3, print = False, train = True)
    #model.plot_sensitivity(train = df.iloc[:int(len(df)*0.7)], test = df.iloc[int(len(df)*0.3):])

    #scaler = StandardScaler()
    #df_padronizado = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    #print(df)

    #model = DecisionTree(data = df, y = "petal_length", max_depth = 4, print = False)
    print(model)
    print(model.ls)
    print(model.rs)
    values = model.predict(df)
    df["y"] = values

    print(df)
    print(model(df.iloc[20:20+1]))
    model.plot_tree()

