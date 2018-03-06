import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn import linear_model

path = ### ADD YOUR PATH HERE
bpd = pd.read_csv(path)

bpd.head()

bike_data = bpd.copy()

# Indexing columns by label
bike_data = bike_data.loc[:, ['cnt','temp']]

bike_data.head()

# Visualization:
bike_data.plot(kind = "scatter",
              x = "temp",
              y = "cnt",
              color = "black")

# Initialize Model
regression = linear_model.LinearRegression()


# Fitting the model
regression.fit(X = pd.DataFrame(bike_data["temp"]), y = bike_data["cnt"])

# Let's look at coefficients

# Y-Intercept
print(regression.intercept_)

# Slope
print(regression.coef_)

regression.score(X = pd.DataFrame(bike_data["temp"]), y = bike_data["cnt"])


bike_data.plot(kind = "scatter",
              x = "temp",
              y = "cnt",
              color = "black")

# Regression Line
plt.plot(bike_data["temp"],
        predictions,
        color = "red")
