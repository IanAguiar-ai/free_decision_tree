import pandas as pd
import matplotlib.pyplot as plt
from time import time
from math import log

def simple_loss(y:pd.DataFrame) -> float:
    y_:float = y.mean()
    return 1/len(y) * sum([(y_i - y_)*(y_i - y_) for y_i in y])

def calc_loss(loss_1:float, loss_2:float) -> float:
    return max(loss_1, loss_2) # loss_1*loss_1, loss_2*loss_2

class Plot:
    __slots__ = ("total", "length", "time", "initial_time", "dif_time", "count")
        
    def __init__(self, total:int, length:int = 10, dif_time:float = 0.2):
        self.total:int = total
        self.length:int = length
        self.dif_time = dif_time
        self.count = 0
        self.time = time()
        self.initial_time = time()

    def load(self, close:bool = False) -> None:
        if self.count == 0:
            self.initial_time = time()

        self.count += 1
        now:int = self.count
        if time() - self.time > self.dif_time or (now == self.total) or close:
            try:
                position:int = int(0.5+now*self.length)//self.total
                _position:int = self.length - (now*self.length)//self.total
                load:str = "|" + "#"*position + "-"*_position + "|"
                seconds_to_end:int = (time() - self.initial_time)/self.count * (self.total - self.count)

                if close:
                    print(f"\r|{'#'*self.length}| {self.total}/{self.total} | 0000 seconds to end")
                    self.count = 0
                elif self.count <= self.total:
                    print(f"\r{load} {now}/{self.total} | {int(seconds_to_end):04} seconds to end", end = "")
            except ZeroDivisionError:
                pass
            self.time = time()
        return None

class DecisionTree:
    """
    ...
    """
    __slots__ = ("dt", "y", "__min_samples", "len_dt", "division", "variable_division", "__depth", "__max_depth", "ls", "rs",
                 "__function_loss", "__calc_loss", "value_loss", "output", "__y_loss", "__args", "__print_",
                 "plot", "__jumps")
    
    def __init__(self, data:pd.DataFrame, y:str, max_depth:int = 3, min_samples:int = 3, *, 
                 loss_function:"function" = simple_loss, loss_calc:"function" = calc_loss,
                 plot:Plot = None, train:bool = True, depth:int = None, print:bool = False, otimized:bool = True) -> None:
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
        self.__depth:int = depth if depth != None else 0
        self.__max_depth:int = max_depth
        self.__print_:bool = print

        # Sons
        self.ls:DecisionTree = None # Left Son
        self.rs:DecisionTree = None # Right Son

        # Train
        self.__function_loss:"function" = loss_function
        self.__calc_loss:"function" = loss_calc
        self.value_loss:float = None
        self.output:float = self.dt[self.y].mean()
        self.__y_loss:float = self.__function_loss(self.dt[self.y])

        # To plot loading
        if plot == None:
            self.plot = Plot(total = int(2**(self.__max_depth+1) - 1), length = 50)
        else:
            self.plot = plot

        # Otimized
        if otimized:
            self.__jumps = max(1, self.len_dt//2_000 * self.__max_depth)
        else:
            self.__jumps = 1

        # For son
        self.__args:dict = {"y":self.y,
                            "min_samples":self.__min_samples,
                            "depth":self.__depth+1,
                            "max_depth":self.__max_depth,
                            "loss_function":self.__function_loss,
                            "loss_calc":self.__calc_loss,
                            "print":self.__print_,
                            "plot":self.plot,
                            "otimized":otimized,
                            "train":False}

        # Train
        if train:
            self.train()

    def __call__(self, X:pd.DataFrame) -> float:
        """
        ...
        """
        return self.predict(X)

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
    Self loss: {self.__y_loss}
    Loss in division: {self.value_loss}

Variables:
    Min Samples: {self.__min_samples}
    Max Depth: {self.__max_depth}

Output: {self.output}
"""

    def __recursive_predict(self, X:pd.DataFrame) -> float:
        """
        ...
        """
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

    def predict(self, X:pd.DataFrame) -> float or list:
        """
        ...
        """
        if len(X) == 1: # float
            return self.__recursive_predict(X)
        else: # list
            return [self.__recursive_predict(X.iloc[i:i+1]) for i in range(len(X))]

    def train(self) -> None:
        """
        ...
        """
        self.plot.load()
        
        self.__print(f"Train:\n\tDepth: {self.__depth} | Lenth: {self.len_dt}")
        if (self.len_dt < 2*self.__min_samples) or (self.__depth >= self.__max_depth):
            return None
        
        for col in self.dt.columns:
            if col != self.y:
                for i in range(0, len(self.dt[col]), self.__jumps):
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

        # To stop print load
        if self.__depth == 0:
            self.plot.load(close = True)
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
        if self.variable_division != None:
            # Division dataframe
            self.ls:DecisionTree = DecisionTree(self.dt[self.dt[self.variable_division] <= self.division], **self.__args)
            self.rs:DecisionTree = DecisionTree(self.dt[self.dt[self.variable_division] > self.division], **self.__args)

            # Recursive train
            self.ls.train()
            self.rs.train()
        return None

    def plot_tree(self, ax = None, x:float = 0.5, y:float = 1.0, dx:float = 0.25, dy:float = 0.12, figsize:tuple = None, fontsize:int = None):
        """
        ...
        """
        if figsize == None:
            figsize = (3.5*self.__max_depth, 2*self.__max_depth)
        
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            ax.set_axis_off()
            self.plot_tree(ax = ax, x = x, y = y, dx = dx, dy = dy, figsize = figsize, fontsize = fontsize)
            plt.tight_layout()
            plt.show()
            return

        # texto do nó
        if self.variable_division is None:
            label = f"Leaf\nSamples: {self.len_dt}\nLoss: {self.__y_loss:.4f}\nOutput: {self.output:.4f}"
            color:str = "lightgreen"
            alpha:float = 1
        elif self.__depth == 0:
            label = f"Output: {self.y}\n{self.variable_division} <= {self.division:.2f}\nSamples: {self.len_dt}\nLoss: {self.__y_loss:.4f}"
            color:str = "orange"
            alpha:float = 1
        else:
            label = f"{self.variable_division} <= {self.division:.2f}\nSamples: {self.len_dt}\nLoss: {self.__y_loss:.4f}"
            color:str = "lightblue"
            alpha:float = 1

        if fontsize == None:
            fontsize = max(15 - 2*self.__depth, 6)

        ax.text(x, y, label, ha = "center", va = "center",
                bbox = dict(boxstyle = "round", facecolor = color, edgecolor = "black", alpha = alpha),
                fontsize = fontsize)

        # Son
        if self.ls is not None:
            ax.plot([x, x-dx], [y-0.02, y-dy+0.02], color = "black")
            self.ls.plot_tree(ax = ax, x = x-dx, y = y-dy, dx = dx/2, dy = dy)
        if self.rs is not None:
            ax.plot([x, x+dx], [y-0.02, y-dy+0.02], color = "black")
            self.rs.plot_tree(ax = ax, x = x+dx, y = y-dy, dx = dx/2, dy = dy)
        return None

    def plot_sensitivity(self, train:pd.DataFrame, test:pd.DataFrame, y = None) -> None:
        """
        ...
        """
        if y == None:
            y:str = self.y
            
        answers:dict = {"depth":[], "mse_train":[], "mse_test":[]}
        for i in range(1, int(log(self.len_dt/self.__min_samples, 2)) + 1):
            answers["depth"].append(i)
            temporary_model:DecisionTree = DecisionTree(train, y = self.y, max_depth = i,
                                                        loss_function = self.__function_loss,
                                                        loss_calc = self.__calc_loss)

            y_:list = temporary_model.predict(train)
            y_:list = 1/len(y_) * sum([(y_real - y_i)*(y_real - y_i) for y_real, y_i in zip(train[y], y_)])
            answers["mse_train"].append(y_)
            
            y_:list = temporary_model.predict(test)
            y_:list = 1/len(y_) * sum([(y_real - y_i)*(y_real - y_i) for y_real, y_i in zip(test[y], y_)])
            answers["mse_test"].append(y_)

        answers:pd.DataFrame = pd.DataFrame(answers)

        best_idx   = answers["mse_test"].idxmin()
        best_depth = int(answers.loc[best_idx, "depth"])

        fig, ax = plt.subplots(figsize = (5, 3))
        plt.plot(answers["depth"], answers["mse_train"], color = "blue", label = "Train")
        plt.plot(answers["depth"], answers["mse_test"], color = "orange", label = "Test")
        plt.axvline(x = best_depth, color = "red", linestyle = "--", linewidth = 1, alpha = 0.8, label = f"Ideal depth: {best_depth}")
        plt.title("Sensitivity Test")
        plt.ylabel("MSE")
        plt.xlabel("Depth")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        return None

    def plot_ci(self, test:pd.DataFrame = None, y:str = None, figsize:tuple = (5, 6), confidence:float = 0.95) -> None:
        """
        ...
        """
        def _confidence(real:list, expected:list, interval:float = 0.95) -> float:
            diferences:list = sorted([abs(real_i - expected_i) for real_i, expected_i in zip(real, expected)])
            return diferences[int((len(diferences)+0.5)*interval)]
            
        if test == None:
            test:pd.DataFrame = self.dt
            
        if y == None:
            y:str = self.y

        y_estimate:list = self.predict(test)
        y_real:list = list(test[y])
        confidence_value:float = _confidence(y_real, y_estimate, interval = confidence)

        expected:list = [min(min(y_estimate), min(y_real)), max(max(y_estimate), max(y_real))]
        ci_1:float = [expected[0]-confidence_value, expected[1]-confidence_value]
        ci_2:float = [expected[0]+confidence_value, expected[1]+confidence_value]
    
        fig, ax = plt.subplots(figsize = (5, 5), zorder = 0)
        plt.scatter(y_real, y_estimate, color = "darkblue", alpha = 1/(len(test))**(1/3), label = "Sample")
        plt.plot(expected, expected,
                 color = "red", linestyle = "--", label = "Expected", zorder = 2)
        plt.plot(expected, ci_1, color = "green", alpha = 0.9, linestyle = "--", zorder = 2)
        plt.plot(expected, ci_2, color = "green", alpha = 0.9, linestyle = "--", zorder = 2)
        plt.fill_between(expected, ci_1, ci_2, color = "green", alpha = 0.2, label = f"Confidence Interval\n({confidence*100}%) (~{confidence_value:0.04f})", zorder = 1)

        plt.xlabel("Real")
        plt.ylabel("Estimate")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        return None
        
