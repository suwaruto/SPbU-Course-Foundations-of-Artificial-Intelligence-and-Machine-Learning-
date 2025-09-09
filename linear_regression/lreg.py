import numpy as np
import sklearn 
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("future.no_silent_downcasting", True)

def parse_csv( filename : str ):
    data = pd.read_csv(filename, index_col = 0)
    data = data.replace(to_replace = {"yes" : 1, "no" : 0})
    data = data.astype(float)
    return data

filename = input("Enter csv-dataset filename (leave empty to use default " +
                "Housing.csv in ./): ")
if not filename:
    df = parse_csv("Housing.csv")
else:
    df = parse_csv(filename)

#prepare training/test data
y = df["price"]
df = df.drop("price", axis = 1)
X = df.to_numpy()
X_lotsize = X[:, 0].reshape(-1, 1)

lreg = sklearn.linear_model.LinearRegression() # model

#train the regression model with all features present
lreg.fit(X, y)
y_predict = lreg.predict(X)

r2 = sklearn.metrics.r2_score(y, y_predict)

plt.title("All features were used for training")
plt.scatter(X_lotsize, y, color = "blue")
plt.scatter(X_lotsize, y_predict, color = "red")
plt.xlabel("lotsize")
plt.ylabel("price")
plt.legend(["Test data", "Predicted data"])
plt.text(0, 0, f"r^2 = {r2}")
plt.savefig("all_features.svg")
plt.show()

#train the regression model with only lotsize feature present
lreg.fit(X_lotsize, y)
y_predict = lreg.predict(X_lotsize)

r2 = sklearn.metrics.r2_score(y, y_predict)

plt.title("Only lotsize feature was used for training")
plt.scatter(X_lotsize, y, color = "blue")
plt.plot(X_lotsize, y_predict, color = "red", lw = 3.0)
plt.xlabel("lotsize")
plt.ylabel("price")
plt.legend(["Test data", "Predicted data"])
plt.text(0, 0, f"r^2 = {r2}")
plt.savefig("lotsize_only.svg")
plt.show()
