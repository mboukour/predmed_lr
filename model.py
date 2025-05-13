import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PredMedLinearRegression:
    """
    A simple linear regression model to predict medical insurance charges based on various features.

    Attributes:
        __x (pd.DataFrame): The feature matrix, including all input features and bias term.
        __y (np.ndarray): The target variable (charges).
        __weights (np.ndarray): The model's weights.
        __learning_rate (float): The learning rate for the gradient descent algorithm.
        __max_epoch (int): The maximum number of training epochs.
        __mse_history (list): History of mean squared errors for each epoch.
        __original_factors (dict): Stores the original mean and standard deviation for each feature column for scaling.
    """

    def __init__(self, filename: str = "insurance.csv", learning_rate: float = 0.1, max_epoch: int = 400):
        """
        Initializes the linear regression model by loading the dataset, performing preprocessing, and setting up the model.
    
        Args:
            filename (str, optional): Path to the CSV file containing the dataset. Defaults to "insurance.csv".
            learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.1.
            max_epoch (int, optional): The maximum number of training epochs. Defaults to 400.
        """
        df = pd.read_csv(filename)
        df["smoker"] = (df["smoker"] == "yes").astype(int)
        df["southwest"] = (df["region"] == "southwest").astype(int)
        df["northwest"] = (df["region"] == "northwest").astype(int)
        df["northeast"] = (df["region"] == "northeast").astype(int)
        df["male"] = (df["sex"] == "male").astype(int)
        df.drop(columns=["region", "sex"], inplace=True)
        self.__x = df.drop('charges', axis=1)
        self.__original_factors = {}
        self.__x["age"] = self.__scale("age")
        self.__x["bmi"] = self.__scale("bmi")
        self.__x["children"] = self.__scale("children")
        self.__x["smoker"] = self.__scale("smoker")
        self.__x["male"] = self.__scale("male")
        self.__x["southwest"] = self.__scale("southwest")
        self.__x["northwest"] = self.__scale("northwest")
        self.__x["northeast"] = self.__scale("northeast")
        self.__x["bias"] = 1
        self.__x.to_numpy()
        self.__y = df["charges"].to_numpy()
        self.__weights = np.zeros(self.__x.shape[1])
        self.__learning_rate = learning_rate
        self.__max_epoch = max_epoch
        self.__mse_history = []

    def __scale_value(self, column: str, value: float) -> float:
        """
        Scales a single feature value using the stored mean and standard deviation.

        Args:
            column (str): The name of the feature.
            value (float): The value to scale.

        Returns:
            float: The scaled value.
        """
        factor = self.__original_factors[column]
        return (value - factor["mean"]) / factor["std"]

    def __scale(self, column_name: str) -> pd.Series:
        """
        Scales the feature values to have a mean of 0 and a standard deviation of 1.

        Args:
            column_name (str): The name of the column to scale.

        Returns:
            pd.Series: The scaled values of the column.
        """
        serie = self.__x[column_name]
        self.__original_factors[column_name] = {"mean": serie.mean(), "std": serie.std()}
        serie = (serie - serie.mean()) / serie.std()
        return serie

    def train(self) -> None:
        """
        Trains the linear regression model using gradient descent to minimize the mean squared error (MSE).
        
        Updates the model's weights during training based on the gradient of the loss function.

        The training stops early if the MSE change between epochs is very small (less than 1e-8), and the learning rate is adjusted accordingly.
        """
        m = self.__x.shape[0]
        for epoch in range(self.__max_epoch):
            predictions = np.dot(self.__x, self.__weights)
            error = self.__y - predictions
            gradient = 2 / m * np.dot(self.__x.T, error)
            self.__weights += self.__learning_rate * gradient
            mse = 1 / m * np.dot(error.T, error)
            if len(self.__mse_history) > 0 and abs(mse - self.__mse_history[-1]) < 1e-8:
                self.__learning_rate *= 0.1
            self.__mse_history.append(mse)


    def predict(self, input_data) -> float:
        """
        Makes a prediction based on input data.

        Args:
            input_data (dict or list): A dictionary or list containing the feature values. The dictionary keys must match the feature names used for training.

        Returns:
            float: The predicted charge based on the input data.
        
        Raises:
            ValueError: If the input data is neither a dictionary nor a list with the correct number of values.
        """
        if isinstance(input_data, dict):
            x_scaled = np.array([
                self.__scale_value("age", input_data.get("age")),
                self.__scale_value("bmi", input_data.get("bmi")),
                self.__scale_value("children", input_data.get("children")),
                self.__scale_value("smoker", input_data.get("smoker")),
                self.__scale_value("male", input_data.get("male")),
                self.__scale_value("southwest", 1 if input_data.get("region") == "southwest" else 0),
                self.__scale_value("northwest", 1 if input_data.get("region") == "northwest" else 0),
                self.__scale_value("northeast", 1 if input_data.get("region") == "northeast" else 0),
                1
            ])
        elif isinstance(input_data, list) and len(input_data) == 8:
            x_scaled = np.array(input_data)
        else:
            raise ValueError("Invalid input format. Expected dict or list.")
        
        return float(np.dot(x_scaled, self.__weights))
    
    def plot_mse(self, save=None) -> None:
        """
        Generates a scatter plot of Mean Squared Error (MSE) for each epoch during training.

        Args:
            save (str, optional): Path to save the plot image. If not specified, the plot will be displayed instead.
        
        Raises:
            ValueError: If 'save' is not a string or None.
        """
        plt.scatter(range(self.__max_epoch), self.__mse_history)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MSE by Epoch")
        if not save:
            plt.show()
        elif not isinstance(save, str):
            raise ValueError("Argument 'save' must be a string or None.")
        else:
            plt.savefig(save)

    def get_mse(self) -> float:
        """
        Returns the final Mean Squared Error (MSE) value after training.
    
        Returns:
            float: The last recorded MSE from training, representing the model's error on the training set.
        """
        return self.__mse_history[-1]

