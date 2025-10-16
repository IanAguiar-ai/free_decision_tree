from time import time
from math import log
from random import seed

# To export and import
import pickle
from pathlib import Path

# Others
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

variables_methods:tuple = ("dt", "y", "X", "__min_samples", "len_dt", "division", "variable_division", "__depth", "__max_depth", "ls", "rs",
                           "__function_loss", "__calc_loss", "value_loss", "output", "__y_loss", "__args", "__print_",
                           "plot", "__jumps", "__dt_with_y", "__tree_search", "__tree_search_w",
                           "__all_trees", "__how_many_trees", "__samples_for_tree", "__seed")

def simple_loss(y:pd.DataFrame) -> float:
    y_:float = y.mean()
    return sum([(y_i - y_)*(y_i - y_) for y_i in y])

def calc_loss(loss_1:float, loss_2:float) -> float:
    return loss_1 + loss_2

def mean(dt:pd.DataFrame):
    return dt.mean()

class Plot:
    """
    ...
    """
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
    __slots__ = variables_methods
    
    def __init__(self, data:pd.DataFrame, y:str, max_depth:int = 3, min_samples:int = 1, *, 
                 loss_function:"function" = simple_loss, loss_calc:"function" = calc_loss, function_prediction_leaf:"function" = mean,
                 plot:Plot = None, train:bool = True, depth:int = None, print:bool = False, tree_search:bool = False, tree_search_w:int = 1, optimized:int = -1) -> None:
        """
        ...
        """
        # Basic information
        self.dt:pd.DataFrame = data
        self.y:str = y
        self.X:list = [col if col != self.y else None for col in self.dt.columns]
        while None in self.X:
            self.X.remove(None)
        self.__min_samples:int = min_samples
        self.len_dt:int = len(data)
        self.division:float = None
        self.variable_division:str = None
        self.__depth:int = depth if depth != None else 0
        self.__max_depth:int = max_depth
        self.__print_:bool = print
        self.__dt_with_y:pd.DataFrame = False # To smoothing tecnique

        # Sons
        self.ls:DecisionTree = None # Left Son
        self.rs:DecisionTree = None # Right Son

        # Train
        self.__function_loss:"function" = loss_function
        self.__calc_loss:"function" = loss_calc
        self.value_loss:float = None
        self.output:float = function_prediction_leaf(self.dt[self.y])
        self.__y_loss:float = self.__function_loss(self.dt[self.y])
        self.__tree_search:bool = tree_search
        self.__tree_search_w:int = tree_search_w

        # To plot loading
        if plot == None:
            self.plot = Plot(total = int(2**(self.__max_depth+1) - 1), length = 50)
        else:
            self.plot = plot

        # optimized
        if (optimized == False) or (optimized == None):
            self.__jumps = 1
        else:
            optimized = self.len_dt//optimized if optimized >= 1 else self.len_dt//2_000 * self.__max_depth
            self.__jumps = max(1, optimized)

        # For son
        self.__args:dict = {"y":self.y,
                            "min_samples":self.__min_samples,
                            "depth":self.__depth+1,
                            "max_depth":self.__max_depth,
                            "loss_function":self.__function_loss,
                            "loss_calc":self.__calc_loss,
                            "function_prediction_leaf":function_prediction_leaf,
                            "print":self.__print_,
                            "plot":self.plot,
                            "optimized":optimized,
                            "train":False,
                            "tree_search":tree_search,
                            "tree_search_w":tree_search_w}

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
    Columns of DataFrame (X): {', '.join((list(self.X)))}
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
    optimized: {'True ' + '(' + str(self.len_dt//self.__jumps) + ' tests per dimension)' if self.__jumps != 1 else 'False'}

Functions:
    Loss Function: {self.__function_loss.__name__}
    Join Loss: {self.__calc_loss.__name__}

Output: {self.output}
"""

    def __recursive_predict(self, X:pd.DataFrame, *, memory_depth = None, which_leaf = None, max_depth = None) -> float:
        """
        ...
        """
        if (max_depth != None) and (max_depth <= memory_depth[0]):
            return self.output
        if (self.division == None) or (self.variable_division == None):
            return self.output

        self.__print(f"Depth: {self.__depth}, Len: {self.len_dt} | {self.division} <= ({self.variable_division})? {X[self.variable_division].iloc[0]}", end = " ")
        if X[self.variable_division].iloc[0] <= self.division:
            self.__print(f"<--")
##            if which_leaf != None:
##                which_leaf[0] += 2**memory_depth[0]*0
            if self.ls != None:
                return self.ls.predict(X, memory_depth = memory_depth, which_leaf = which_leaf, max_depth = max_depth)
            else:
                return self.output
        else:
            self.__print(f"-->")
            if which_leaf != None:
                which_leaf[0] += 2**memory_depth[0]
            if self.rs != None:
                return self.rs.predict(X, memory_depth = memory_depth, which_leaf = which_leaf, max_depth = max_depth)
            else:
                return self.output

    def predict(self, X:pd.DataFrame, *, memory_depth = None, which_leaf = None, max_depth:int = None) -> float or list:
        """
        Predict outputs for one or multiple samples.

        Args:
            X (pd.DataFrame): Input features. Can be one row or multiple rows.

        Returns:
            float or list: Predicted value for a single row or list of values for multiple rows.
        """
        if memory_depth != None:
            memory_depth[0] += 1

        if len(X) == 1: # float
            return self.__recursive_predict(X, memory_depth = memory_depth, which_leaf = which_leaf, max_depth = max_depth)
        else: # list
            return [self.__recursive_predict(X.iloc[i:i+1], memory_depth = [0] if memory_depth == None else memory_depth,
                                             which_leaf = which_leaf, max_depth = max_depth) for i in range(len(X))]

    def predict_smooth(self, X:pd.DataFrame, n_neighbors:int = None, alpha:float = 0.00001, beta:float = 2, representatives:bool = True) -> float or list:
        """
        Predict using a smoothed approximation based on nearest neighbors of tree leaves.

        Args:
            X (pd.DataFrame): Input features.
            n_neighbors (int, optional): Number of neighbors to consider. Defaults to all.
            alpha (float, optional): Small constant to avoid division by zero in weights.
            beta (float, optional): Exponent applied to distances in weight calculation.
            representatives (bool, optional): If True, use mean representatives of each leaf.

        Returns:
            float or list: Smoothed prediction for one or multiple samples.
        """
        if (type(self.__dt_with_y) == bool) and (self.__dt_with_y == False):
            self.__dt_with_y:pd.DataFrame = self.detect_depth()
            if representatives:
                self.__dt_with_y:pd.DataFrame = self.__dt_with_y[[*self.X, "__dt_y__", "__dt_leaf__"]].groupby("__dt_leaf__").mean()
            self.__dt_with_y:pd.DataFrame = self.__dt_with_y.reset_index(drop = True)

        results:list = []
        n_neighbors:int = len(self.__dt_with_y) if n_neighbors == None else n_neighbors #len(self.X) + 1 if n_neighbors == None else n_neighbors
        for i in range(len(X)):
            line_temporary = X[self.X].iloc[i]
            
            distances:float = np.linalg.norm(self.__dt_with_y[self.X].values - line_temporary.values, axis = 1)
            nearest_indices = np.argsort(distances)[:n_neighbors]
            n_distances:list = distances[nearest_indices]
            
            weights:list = [1/(alpha+distance**beta) for distance in n_distances]
            weights:list = [w/sum(weights) for w in weights]

            results.append(sum([self.__dt_with_y.iloc[k]["__dt_y__"]*weights[index] for index, k in enumerate(nearest_indices)]))
            
        return results[0] if len(results) == 1 else results

    def train(self) -> None:
        """
        Train the decision tree recursively by splitting nodes.
        """
        self.plot.load()
        self.__dt_with_y:pd.DataFrame = False # To smoothing tecnique
        
        self.__print(f"Train:\n\tDepth: {self.__depth} | Lenth: {self.len_dt}")
        if (self.len_dt < 2*self.__min_samples) or (self.__depth >= self.__max_depth):
            return None
        
        for col in self.dt.columns:
            if col != self.y:
                if self.__tree_search:
                    w_tree:int = self.__tree_search_w
                    self.dt = self.dt.sort_values(by = col)
                    i_min, i_now, i_max = 0, len(self.dt)//2, len(self.dt) - 2                
                    while True:
                        division_min:int = self.dt.iloc[i_min][col]
                        dt_1_min:pd.DataFrame = self.dt[self.dt[col] <= division_min]
                        dt_2_min:pd.DataFrame = self.dt[self.dt[col] > division_min]
                        
                        final_loss_min:float = self.__calc_loss_tree(dt_1_min, dt_2_min, col)

                        division_max:int = self.dt.iloc[i_max][col]
                        dt_1_max:pd.DataFrame = self.dt[self.dt[col] <= division_max]
                        dt_2_max:pd.DataFrame = self.dt[self.dt[col] > division_max]
                        
                        final_loss_max:float = self.__calc_loss_tree(dt_1_max, dt_2_max, col)

                        if (i_min == i_now) or (i_max == i_now):
                            self.__update_parameters(division = division_min, variable = col, loss = final_loss_min)
                            break
                        
                        if min(final_loss_min, final_loss_max) == final_loss_min:
                            self.__update_parameters(division = division_min, variable = col, loss = final_loss_min)
                            i_max = int((i_now + i_max*w_tree)//(w_tree+1))
                            i_now = (i_max + i_min)//2

                        elif min(final_loss_min, final_loss_max) == final_loss_max:
                            self.__update_parameters(division = division_max, variable = col, loss = final_loss_max)
                            i_min = int((i_now + i_min*w_tree)//(w_tree+1) + 1)
                            i_now = (i_max + i_min)//2

                else:
                    #self.dt = self.dt.sort_values(by = col)
                    #all_loses = []
                    for i in range(0, len(self.dt[col]), self.__jumps):
                        division:int = self.dt.iloc[i][col]
                        dt_1:pd.DataFrame = self.dt[self.dt[col] <= division]
                        dt_2:pd.DataFrame = self.dt[self.dt[col] > division]

                        if (len(dt_1) >= self.__min_samples) and (len(dt_2) >= self.__min_samples):
                            final_loss:float = self.__calc_loss_tree(dt_1, dt_2, col)
                            #all_loses.append(final_loss)
                            
                            if (self.division == None) or (self.variable_division == None):
                                self.__update_parameters(division = division, variable = col, loss = final_loss)
                            elif final_loss < self.value_loss:
                                self.__update_parameters(division = division, variable = col, loss = final_loss)
                    #plt.figure(figsize = (10, 6))
                    #plt.plot([i for i in range(len(all_loses))], all_loses)
                    #plt.grid()
                    #plt.show()

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
        if (self.value_loss == None) or (self.value_loss > loss):
            self.division:float = float(division)
            self.variable_division:str = str(variable)
            self.value_loss:float = float(loss)
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

    def detect_depth(self) -> pd.DataFrame:
        """
        ...
        """
        temporary_depth:list = []
        temporary_leaf:list = []
        for i in range(len(self.dt)):
            memory_depth:list = [-1]
            which_leaf:list = [0]
            self.predict(self.dt.iloc[i:i+1], memory_depth = memory_depth, which_leaf = which_leaf)
            temporary_depth.append(memory_depth[0])
            temporary_leaf.append(which_leaf[0])

        df_temporary:pd.DataFrame = self.dt.copy()
        df_temporary["__dt_depth__"] = temporary_depth
        df_temporary["__dt_leaf__"] = temporary_leaf
        df_temporary["__dt_y__"] = self.predict(self.dt)
        return df_temporary

    def save(self, path:str) -> None:
        """
        Saves the complete tree.
        """
        p = Path(str(path) + ".decisiontree")
        p.parent.mkdir(parents = True, exist_ok = True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol = pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str) -> "DecisionTree":
        """
        Loads and returns a tree previously saved by DecisionTree.save().
        """
        if not ".decisiontree" in path:
            path = str(path) + ".decisiontree"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("The file does not contain a DecisionTree instance.")
        return obj

    def plot_tree(self, ax = None, x:float = 0.5, y:float = 1.0, dx:float = 0.25, dy:float = 0.12, figsize:tuple = None, fontsize:int = None):
        """
        Plot the tree structure.

        Args:
            ax (matplotlib.axes.Axes, optional): Existing axis to draw on.
            x (float): Horizontal position of the current node.
            y (float): Vertical position of the current node.
            dx (float): Horizontal distance between parent and children.
            dy (float): Vertical distance between levels.
            figsize (tuple, optional): Figure size if a new plot is created.
            fontsize (int, optional): Font size for node labels.

        Returns:
            None
        """
        if figsize == None:
            figsize = (3 + 2**(self.__max_depth), 2*self.__max_depth)
        
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

    def plot_sensitivity(self, train:pd.DataFrame, test:pd.DataFrame, y = None) -> int:
        """
        Evaluate tree depth sensitivity by measuring MSE on train and test sets.

        Args:
            train (pd.DataFrame): Training dataset.
            test (pd.DataFrame): Testing dataset.
            y (str, optional): Target column. Defaults to self.y.

        Returns:
            int: Best depth (minimum MSE on test set).
            """
        if y == None:
            y:str = self.y

        answers:dict = {"depth":[], "mse_train":[], "mse_test":[]}
        
        max_depth:int = int(log(self.len_dt/self.__min_samples, 2)) + 1
        
        args:dict = {"data":train,
                     "y":self.y,
                     "min_samples":self.__min_samples,
                     "loss_function":self.__function_loss,
                     "loss_calc":self.__calc_loss,
                     "tree_search":self.__tree_search,
                     "tree_search_w":self.__tree_search_w}
        
        temporary_model:DecisionTree = DecisionTree(max_depth = max_depth, **args)
        for i in range(1, max_depth):
            answers["depth"].append(i)

            y_:list = temporary_model.predict(train, max_depth = i)
            y_:list = 1/len(y_) * sum([(y_real - y_i)*(y_real - y_i) for y_real, y_i in zip(train[y], y_)])
            answers["mse_train"].append(y_)
            
            y_:list = temporary_model.predict(test, max_depth = i)
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
        return best_depth

    def plot_ci(self, test:pd.DataFrame = False, y:str = None, figsize:tuple = (5, 6), confidence:float = 0.95) -> float:
        """
        Plot confidence interval of model predictions.

        Args:
            test (pd.DataFrame, optional): Test dataset. Defaults to training data.
            y (str, optional): Target column. Defaults to self.y.
            figsize (tuple, optional): Figure size.
            confidence (float, optional): Confidence level (0-1). Defaults to 0.95.

        Returns:
            float: Confidence value (approximate error bound).
        """
        def _confidence(real:list, expected:list, interval:float = 0.95) -> float:
            diferences:list = sorted([abs(real_i - expected_i) for real_i, expected_i in zip(real, expected)])
            return diferences[int((len(diferences)+0.5)*interval)]
            
        if (type(test) == bool) and (test == False):
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

        plt.xlabel(f"Real ({self.y})")
        plt.ylabel(f"Estimate ({self.y})")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        return confidence_value

class RandomFlorest:
    """
    ...
    """
    __slots__ = variables_methods
    
    def __init__(self, data:pd.DataFrame, y:str, max_depth:int = 4, min_samples:int = 1, how_many_trees:int = 10, samples_for_tree:int = None, seed:int = 1, *, 
                 loss_function:"function" = simple_loss, loss_calc:"function" = calc_loss, function_prediction_leaf:"function" = mean,
                 plot:Plot = None, train:bool = True, depth:int = None, print:bool = False, tree_search:bool = False, tree_search_w:int = 1, optimized:int = -1) -> None:
        """
        ...
        """
        # Basic information
        self.dt:pd.DataFrame = data
        self.y:str = y
        self.X:list = [col if col != self.y else None for col in self.dt.columns]
        while None in self.X:
            self.X.remove(None)
        self.__min_samples:int = min_samples
        self.len_dt:int = len(data)
        self.division:float = None
        self.variable_division:str = None
        self.__depth:int = depth if depth != None else 0
        self.__max_depth:int = max_depth
        self.__print_:bool = print
        self.__dt_with_y:pd.DataFrame = False # To smoothing tecnique
        self.__all_trees:list = []
        self.__how_many_trees:int = how_many_trees
        self.__samples_for_tree:int = max((self.len_dt//self.__how_many_trees)*3 + 1, min(20, self.len_dt)) if samples_for_tree == None else samples_for_tree

        # Train
        self.__function_loss:"function" = loss_function
        self.__calc_loss:"function" = loss_calc
        self.value_loss:float = None
        self.output:float = function_prediction_leaf(self.dt[self.y])
        self.__y_loss:float = self.__function_loss(self.dt[self.y])
        self.__tree_search:bool = tree_search
        self.__tree_search_w:int = tree_search_w
        self.__seed:int = seed

        # To plot loading
        if plot == None:
            self.plot = Plot(total = int(2**(self.__max_depth+1) - 1), length = 50)
        else:
            self.plot = plot

        # optimized
        if (optimized == False) or (optimized == None):
            self.__jumps = 1
        else:
            optimized = self.len_dt//optimized if optimized >= 1 else self.len_dt//2_000 * self.__max_depth
            self.__jumps = max(1, optimized)

        # For son
        self.__args:dict = {"y":self.y,
                            "min_samples":self.__min_samples,
                            "max_depth":self.__max_depth,
                            "loss_function":self.__function_loss,
                            "loss_calc":self.__calc_loss,
                            "function_prediction_leaf":function_prediction_leaf,
                            "print":self.__print_,
                            "plot":self.plot,
                            "optimized":optimized,
                            "train":True,
                            "tree_search":tree_search,
                            "tree_search_w":tree_search_w}

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
    Columns of DataFrame (X): {', '.join((list(self.X)))}
    y: {self.y}
    Len of Dataframe: {self.len_dt}
    Number of trees: {self.__how_many_trees}
    Samples per tree: {self.__samples_for_tree}

Variables:
    Min Samples: {self.__min_samples}
    Max Depth: {self.__max_depth}
    optimized: {'True ' + '(' + str(self.len_dt//self.__jumps) + ' tests per dimension)' if self.__jumps != 1 else 'False'}

Functions:
    Loss Function: {self.__function_loss.__name__}
    Join Loss: {self.__calc_loss.__name__}

Output: {self.output}
"""

    def train(self):
        """
        ...
        """
        for _ in range(self.__how_many_trees):
            temporary_data:pd.DataFrame = self.dt.sample(n = self.__samples_for_tree, random_state = self.__seed + _, replace = True)
            self.__all_trees.append(DecisionTree(temporary_data, **self.__args))

    def predict(self, X:pd.DataFrame, edges:int = None, max_depth:int = None) -> list or float:
        """
        ...
        """
        if edges == None:
            values = [0 for i in range(len(X))]
            for _ in range(self.__how_many_trees):
                temporary:list = self.__all_trees[_].predict(X, max_depth = max_depth)
                if type(temporary) != list:
                    temporary:list = [temporary]
                for index, temp in enumerate(temporary):
                    values[index] += temp/self.__how_many_trees
        else:
            values = [[] for i in range(len(X))]
            for _ in range(self.__how_many_trees):
                temporary:list = self.__all_trees[_].predict(X)
                if type(temporary) != list:
                    temporary:list = [temporary]
                for index, temp in enumerate(temporary):
                    values[index].append(temp)

            for index in range(len(values)):
                cut:int = min(max(int(edges*len(values[index])), 1), self.__how_many_trees)
                values[index] = sorted(values[index])

                mean:float = sum(values[index])/len(values[index])
                if abs(sum(values[index][-cut:])/len(values[index][-cut:]) - mean) > abs(sum(values[index][:cut])/len(values[index][:cut]) - mean):
                    values[index] = sum(values[index][:cut])/len(values[index][:cut])
                else:
                    values[index] = sum(values[index][-cut:])/len(values[index][-cut:])

                #values[index] = [*values[index][:cut], *values[index][-cut:]]
                #values[index] = sum(values[index])/len(values[index])

        return values if len(values) > 1 else values[0]

    def save(self, path:str) -> None:
        """
        Saves the complete tree.
        """
        p = Path(str(path) + ".randomflorest")
        p.parent.mkdir(parents = True, exist_ok = True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol = pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str) -> "RandomFlorest":
        """
        Loads and returns a tree previously saved by RandomFlorest.save().
        """
        if not ".randomflorest" in path:
            path = str(path) + ".randomflorest"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("The file does not contain a RandomFlorest instance.")
        return obj

    def plot_sensitivity(self, train:pd.DataFrame, test:pd.DataFrame, y = None, return_dataframe:bool = False) -> (int, int):
        """
        Evaluate tree depth sensitivity by measuring MSE on train and test sets.

        Args:
            train (pd.DataFrame): Training dataset.
            test (pd.DataFrame): Testing dataset.
            y (str, optional): Target column. Defaults to self.y.

        Returns:
            int: Best depth (minimum MSE on test set).
            """
        if y == None:
            y:str = self.y

        answers:dict = {"depth":[], "how_many_trees":[], "mse_train":[], "mse_test":[]}
        
        max_depth:int = int(log(self.len_dt/self.__min_samples, 2)) + 1
        
        args:dict = {"data":train,
                     "y":self.y,
                     "min_samples":self.__min_samples,
                     "loss_function":self.__function_loss,
                     "loss_calc":self.__calc_loss,
                     "tree_search":self.__tree_search,
                     "tree_search_w":self.__tree_search_w,
                     "samples_for_tree":self.__samples_for_tree}

        for how_m_t in [5, 10, 20, 50]:
            temporary_model:RandomFlorest = RandomFlorest(max_depth = max_depth, how_many_trees = how_m_t, **args)
            for i in range(1, max_depth):
                answers["depth"].append(i)
                answers["how_many_trees"].append(how_m_t)

                y_:list = temporary_model.predict(train, max_depth = i)
                y_:list = 1/len(y_) * sum([(y_real - y_i)*(y_real - y_i) for y_real, y_i in zip(train[y], y_)])
                answers["mse_train"].append(y_)
                
                y_:list = temporary_model.predict(test, max_depth = i)
                y_:list = 1/len(y_) * sum([(y_real - y_i)*(y_real - y_i) for y_real, y_i in zip(test[y], y_)])
                answers["mse_test"].append(y_)

        answers:pd.DataFrame = pd.DataFrame(answers)

        best_idx   = answers["mse_test"].idxmin()
        best_depth = int(answers.loc[best_idx, "depth"])
        best_how_many_trees = int(answers.loc[best_idx, "how_many_trees"])


        fig, ax = plt.subplots(figsize = (8, 4))
        plt.scatter(answers["depth"], answers["mse_train"], c = answers["how_many_trees"], marker = "+", alpha = 0.2, label = "Train")
        plt.scatter(answers["depth"], answers["mse_test"], c = answers["how_many_trees"], alpha = 0.2, label = "Test")
        plt.axvline(x = best_depth, color = "red", linestyle = "--", linewidth = 1, alpha = 0.8, label = f"Ideal depth: {best_depth}")
        plt.title(f"Sensitivity Test\nIdeal depth: {best_depth}, Ideal number of trees: {best_how_many_trees}")
        plt.ylabel("MSE")
        plt.xlabel("Depth")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        if return_dataframe:
            return answers
        else:
            return best_depth, best_how_many_trees

class IsolationDecisionTree:
    """
    ...
    """
    __slots__ = variables_methods
    
    def __init__(self, data:pd.DataFrame, max_depth:int = 4, min_samples:int = 1, *, 
                 loss_function:"function" = simple_loss, loss_calc:"function" = calc_loss, function_prediction_leaf:"function" = mean,
                 plot:Plot = None, train:bool = True, depth:int = None, print:bool = False) -> None:
        """
        ...
        """
        # Basic information
        self.dt:pd.DataFrame = data
        self.X:str = list(self.dt.columns)
        self.__min_samples:int = min_samples
        self.len_dt:int = len(data)
        self.division:float = None
        self.variable_division:str = None
        self.__depth:int = depth if depth != None else 0
        self.__max_depth:int = max_depth
        self.__print_:bool = print
        self.__dt_with_y:pd.DataFrame = False # To smoothing tecnique

        # Sons
        self.ls:DecisionTree = None # Left Son
        self.rs:DecisionTree = None # Right Son

        # Train
        self.__function_loss:"function" = loss_function
        self.__calc_loss:"function" = loss_calc
        self.value_loss:float = None
        self.output:float = function_prediction_leaf(self.dt[self.X[0]])

        # To plot loading
        if plot == None:
            self.plot = Plot(total = int(2**(self.__max_depth+1) - 1), length = 50)
        else:
            self.plot = plot

        # For son
        self.__args:dict = {"min_samples":self.__min_samples,
                            "depth":self.__depth+1,
                            "max_depth":self.__max_depth,
                            "loss_function":self.__function_loss,
                            "loss_calc":self.__calc_loss,
                            "function_prediction_leaf":function_prediction_leaf,
                            "print":self.__print_,
                            "plot":self.plot,
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
    Columns of DataFrame (X): {', '.join((list(self.X)))}
    Len of Dataframe: {self.len_dt}

Values:
    Depth: {'root' if self.__depth == 0 else self.__depth}
    Division in: {self.variable_division}
    In value: <= {self.division}
    Loss in division: {self.value_loss}

Variables:
    Min Samples: {self.__min_samples}
    Max Depth: {self.__max_depth}

Functions:
    Loss Function: {self.__function_loss.__name__}
    Join Loss: {self.__calc_loss.__name__}

Output: {self.output}
"""

    def train(self) -> None:
        """
        Train the decision tree recursively by splitting nodes.
        """
        self.plot.load()
        self.__dt_with_y:pd.DataFrame = False # To smoothing tecnique
        
        self.__print(f"Train:\n\tDepth: {self.__depth} | Lenth: {self.len_dt}")
        if (self.len_dt < 2*self.__min_samples) or (self.__depth >= self.__max_depth):
            return None

        all_std:list = sorted([[col, self.dt[col].std()] for col in self.dt.columns], key = lambda x:x[1]) # __calc_loss_tree not necessary
        col_to_cut:str = all_std[-1][0]
        std:float = all_std[-1][1]
        division:int = self.dt[col_to_cut].mean()
        self.__update_parameters(division, variable = col_to_cut, loss = std) 

        # Update tree
        self.__update_tree()

        # To stop print load
        if self.__depth == 0:
            self.plot.load(close = True)
        return None
    def __update_parameters(self, division:float, variable:str, loss:float) -> None:
        """
        ...
        """
        if (self.value_loss == None) or (self.value_loss > loss):
            self.division:float = float(division)
            self.variable_division:str = str(variable)
            self.value_loss:float = float(loss)
        return None

    def __update_tree(self) -> None:
        """
        ...
        """
        if self.variable_division != None:
            # Division dataframe
            self.ls:DecisionTree = IsolationDecisionTree(self.dt[self.dt[self.variable_division] <= self.division], **self.__args)
            self.rs:DecisionTree = IsolationDecisionTree(self.dt[self.dt[self.variable_division] > self.division], **self.__args)

            # Recursive train
            self.ls.train()
            self.rs.train()
        return None

    def detect_depth(self) -> pd.DataFrame:
        """
        ...
        """
        temporary_depth:list = []
        temporary_leaf:list = []
        for i in range(len(self.dt)):
            memory_depth:list = [-1]
            which_leaf:list = [0]
            self.predict(self.dt.iloc[i:i+1], memory_depth = memory_depth, which_leaf = which_leaf)
            temporary_depth.append(memory_depth[0])
            temporary_leaf.append(which_leaf[0])

        df_temporary:pd.DataFrame = self.dt.copy()
        df_temporary["__dt_depth__"] = temporary_depth
        df_temporary["__dt_leaf__"] = temporary_leaf
        df_temporary["__dt_y__"] = self.predict(self.dt)
        return df_temporary

    def isolate(self, threshold:int = 1) -> pd.DataFrame:
        """
        ...
        """
        df_temporary:pd.DataFrame = self.detect_depth()
        df_temporary["__id__"] = df_temporary["__dt_leaf__"]
        df_temporary:pd.DataFrame = df_temporary.groupby("__dt_leaf__").count().sort_values("__dt_depth__")
        df_temporary:pd.DataFrame = df_temporary.reset_index()
        leafs_isolated:list = []
        i:int = 0
        while i < len(df_temporary):
            print(df_temporary.iloc[i]["__dt_depth__"], threshold)
            if df_temporary.iloc[i]["__dt_depth__"] <= threshold:
                leafs_isolated.append(df_temporary.iloc[i]["__dt_leaf__"])
            else:
                break
            i += 1
        temp:pd.DataFrame = self.detect_depth()
        temp:pd.DataFrame = temp[temp["__dt_leaf__"].isin(leafs_isolated)].drop(columns = ["__dt_y__"])
        return temp.reset_index()

    def __recursive_predict(self, X:pd.DataFrame, *, memory_depth = None, which_leaf = None, max_depth = None) -> float:
        """
        ...
        """
        if (max_depth != None) and (max_depth <= memory_depth[0]):
            return self.output
        if (self.division == None) or (self.variable_division == None):
            return self.output

        self.__print(f"Depth: {self.__depth}, Len: {self.len_dt} | {self.division} <= ({self.variable_division})? {X[self.variable_division].iloc[0]}", end = " ")
        if X[self.variable_division].iloc[0] <= self.division:
            self.__print(f"<--")
##            if which_leaf != None:
##                which_leaf[0] += 2**memory_depth[0]*0
            if self.ls != None:
                return self.ls.predict(X, memory_depth = memory_depth, which_leaf = which_leaf, max_depth = max_depth)
            else:
                return self.output
        else:
            self.__print(f"-->")
            if which_leaf != None:
                which_leaf[0] += 2**memory_depth[0]
            if self.rs != None:
                return self.rs.predict(X, memory_depth = memory_depth, which_leaf = which_leaf, max_depth = max_depth)
            else:
                return self.output


    def predict(self, X:pd.DataFrame, *, memory_depth = None, which_leaf = None, max_depth:int = None) -> float or list:
        """
        Predict outputs for one or multiple samples.

        Args:
            X (pd.DataFrame): Input features. Can be one row or multiple rows.

        Returns:
            float or list: Predicted value for a single row or list of values for multiple rows.
        """
        if memory_depth != None:
            memory_depth[0] += 1

        if len(X) == 1: # float
            return self.__recursive_predict(X, memory_depth = memory_depth, which_leaf = which_leaf, max_depth = max_depth)
        else: # list
            return [self.__recursive_predict(X.iloc[i:i+1], memory_depth = [0] if memory_depth == None else memory_depth,
                                             which_leaf = which_leaf, max_depth = max_depth) for i in range(len(X))]

    def save(self, path:str) -> None:
        """
        Saves the complete tree.
        """
        p = Path(str(path) + ".decisiontree")
        p.parent.mkdir(parents = True, exist_ok = True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol = pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str) -> "DecisionTree":
        """
        Loads and returns a tree previously saved by DecisionTree.save().
        """
        if not ".decisiontree" in path:
            path = str(path) + ".decisiontree"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("The file does not contain a DecisionTree instance.")
        return obj

    def plot_isolation(self, dims = None, isolated = None, figsize = (6, 6), max_depth = None, line_kwargs = None):
        """
        ...
        """
        if dims is None:
            dims = [self.X[0], self.X[1]]
        else:
            assert len(dims) == 2, f"lenth of dims has be 2 no {len(dims)}!"

        d1, d2 = dims
        df = self.dt

        plt.figure(figsize=figsize)
        plt.scatter(df[d1], df[d2], s = 30, alpha = 1/(len(self.dt)**(0.2)))

        if isolated is not None:
            isolated = np.asarray(isolated)
            if isolated.dtype == bool:
                mask = isolated
            else:
                mask = np.zeros(len(df), dtype = bool)
                mask[isolated] = True
            if mask.any():
                plt.scatter(df.loc[mask, d1], df.loc[mask, d2], color = "red", s = 40)

        x1, x2 = float(df[d1].min()), float(df[d1].max())
        y1, y2 = float(df[d2].min()), float(df[d2].max())

        if line_kwargs is None:
            line_kwargs = {"linestyle": "--", "color": "red", "linewidth": 0.8, "alpha": 0.7}

        all_cols = list(df.columns)

        def _feat_index(var_name_or_idx):
            if var_name_or_idx == d1: return 0
            if var_name_or_idx == d2: return 1
            if isinstance(var_name_or_idx, (int, np.integer)):
                try:
                    col_name = self.X[var_name_or_idx] if hasattr(self, "X") else all_cols[int(var_name_or_idx)]
                    if col_name == d1: return 0
                    if col_name == d2: return 1
                except Exception:
                    pass
            if isinstance(var_name_or_idx, str):
                if var_name_or_idx in all_cols:
                    return -1
            return -1

        def _draw_splits(node, rx1, rx2, ry1, ry2, depth):
            if node is None:
                return
            if (getattr(node, "ls", None) is None) and (getattr(node, "rs", None) is None):
                return
            if (max_depth is not None) and (depth > max_depth):
                return

            feat = getattr(node, "variable_division", None)
            thr  = getattr(node, "division", None)
            if (feat is None) or (thr is None):
                _draw_splits(getattr(node, "ls", None), rx1, rx2, ry1, ry2, depth+1)
                _draw_splits(getattr(node, "rs", None), rx1, rx2, ry1, ry2, depth+1)
                return

            fidx = _feat_index(feat)

            if fidx == 0:
                x = float(thr)
                if rx1 < x < rx2:
                    plt.plot([x, x], [ry1, ry2], **line_kwargs)
                _draw_splits(getattr(node, "ls", None), rx1, min(rx2, x), ry1, ry2, depth+1)
                _draw_splits(getattr(node, "rs", None), max(rx1, x), rx2, ry1, ry2, depth+1)

            elif fidx == 1:
                y = float(thr)
                if ry1 < y < ry2:
                    plt.plot([rx1, rx2], [y, y], **line_kwargs)
                _draw_splits(getattr(node, "ls", None), rx1, rx2, ry1, min(ry2, y), depth+1)
                _draw_splits(getattr(node, "rs", None), rx1, rx2, max(ry1, y), ry2, depth+1)

            else:
                _draw_splits(getattr(node, "ls", None), rx1, rx2, ry1, ry2, depth+1)
                _draw_splits(getattr(node, "rs", None), rx1, rx2, ry1, ry2, depth+1)

        _draw_splits(self, x1, x2, y1, y2, depth=0)

        plt.xlabel(str(d1)); plt.ylabel(str(d2))
        plt.title(f"Isolation Decision Tree")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
