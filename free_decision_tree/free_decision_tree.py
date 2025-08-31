import pandas as pd

def simple_loss(y:pd.DataFrame) -> float:
    y_:float = y.mean()
    return 1/len(y) * sum([(y_i - y_)*(y_i - y_) for y_i in y])

def calc_loss(loss_1:float, loss_2:float) -> float:
    return loss_1 + loss_2

class DecisionTree:
    """
    ...
    """
    __slots__ = ("dt", "y", "__min_samples", "len_dt", "division", "variable_division", "__depth", "__max_depth", "ls", "rs",
                 "__function_loss", "__calc_loss", "value_loss", "output", "__args", "__print_")
    
    def __init__(self, data:pd.DataFrame, y:str, min_samples:int = 3, depth:int = 0, max_depth:int = 3,
                 loss_function:"function" = simple_loss, loss_calc:"function" = calc_loss, print:bool = False) -> None:
        """
        ...
        """
        # Basic information
        self.dt:pd.DataFrame = data
        self.y:str = y
        self.__min_samples:int = min_samples
        self.len_dt:int = len(data)
        self.division:float = None
        self.variable_division:str = None
        self.__depth:int = depth
        self.__max_depth:int =  max_depth
        self.__print_:bool = print

        # Sons
        self.ls:DecisionTree = None # Left Son
        self.rs:DecisionTree = None # Right Son

        # Train
        self.__function_loss:"function" = loss_function
        self.__calc_loss:"function" = loss_calc
        self.value_loss:float = None
        self.output:float = self.dt[self.y].mean()

        # For son
        self.__args:dict = {"y":self.y,
                            "min_samples":self.__min_samples,
                            "depth":self.__depth+1,
                            "max_depth":self.__max_depth,
                            "loss_function":self.__function_loss,
                            "loss_calc":self.__calc_loss,
                            "print":self.__print_}

    def __print(self, str_:str, end = "\n"):
        if self.__print_:
            print(str_, end = end)
        return None

    def __repr__(self):
        return f"""
DataFrame:
    Columns of DataFrame: {', '.join((list(self.dt.columns)))}
    y: {self.y}
    Len of Dataframe: {self.len_dt}

Values:
    Depth: {'root' if self.__depth == 0 else self.__depth}
    Division in: {self.variable_division}
    In value: <= {self.division}
    Loss: {self.value_loss}

Variables:
    Min Samples: {self.__min_samples}
    Max Depth: {self.__max_depth}

Output: {self.output}
"""

    def predict(self, X:pd.DataFrame) -> float:        
        if (self.division == None) or (self.variable_division == None):
            return self.output

        self.__print(f"Depth: {self.__depth}, Len: {self.len_dt} | {self.division} <= ({self.variable_division})? {X[self.variable_division].iloc[0]}", end = " ")
        if X[self.variable_division].iloc[0] <= self.division:
            self.__print(f"<--")
            if self.ls != None:
                return self.ls.predict(X)
            else:
                return self.output
        else:
            self.__print(f"-->")
            if self.rs != None:
                return self.rs.predict(X)
            else:
                return self.output

    def train(self) -> None:
        """
        ...
        """
        self.__print(f"Train:\n\tDepth: {self.__depth} | Lenth: {self.len_dt}")
        if (self.len_dt < 2*self.__min_samples) or (self.__depth >= self.__max_depth):
            return None
        
        for col in self.dt.columns:
            if col != self.y:
                for i in range(len(self.dt[col])):
                    division:int = self.dt.iloc[i][col]
                    dt_1:pd.DataFrame = self.dt[self.dt[col] <= division]
                    dt_2:pd.DataFrame = self.dt[self.dt[col] > division]

                    if (len(dt_1) >= self.__min_samples) and (len(dt_2) >= self.__min_samples):
                        if (self.division == None) or (self.variable_division == None):
                            final_loss:float = self.__calc_loss_tree(dt_1, dt_2, col)
                            self.__update_parameters(division = division, variable = col, loss = final_loss)

                        else:
                            final_loss:float = self.__calc_loss_tree(dt_1, dt_2, col)
                            
                            if final_loss < self.value_loss:
                                self.__update_parameters(division = division, variable = col, loss = final_loss)

        # Update tree
        self.__update_tree()

        # Recursive train
        self.ls.train()
        self.rs.train()
        return None

    def __calc_loss_tree(self, dt_1:pd.DataFrame, dt_2:pd.DataFrame, col:str) -> float:
        """
        ...
        """
        loss_1:float = self.__function_loss(dt_1[self.y])
        loss_2:float = self.__function_loss(dt_2[self.y])
        return self.__calc_loss(loss_1, loss_2)

    def __update_parameters(self, division:float, variable:str, loss:float) -> None:
        """
        ...
        """
        self.division:float = division
        self.variable_division:str = variable
        self.value_loss:float = loss
        return None

    def __update_tree(self) -> None:
        """
        ...
        """
        self.ls:DecisionTree = DecisionTree(self.dt[self.dt[self.variable_division] <= self.division], **self.__args)
        self.rs:DecisionTree = DecisionTree(self.dt[self.dt[self.variable_division] > self.division], **self.__args)
        return None

if __name__ == "__main__":
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    df = sns.load_dataset("iris")  # ou "tips", "titanic", "penguins", etc.
    df = df.drop(columns = ["species"])

    scaler = StandardScaler()
    df_padronizado = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


    print(df)

    model = DecisionTree(data = df, y = "petal_length", print = False)
    model.train()
    print(model.ls)
    print(model.rs)
    values = []
    for i in range(len(df)):
        #print(f"i: {i:03}")
        values.append(model.predict(df.iloc[i:i+1]))

    df["y"] = values

    print(df)
    print(model.predict(df.iloc[20:20+1]))
