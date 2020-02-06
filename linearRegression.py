# creating our own linear regression algorithm
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("fivethirtyeight")


# xs = np.array([1, 2, 3, 4, 5, 6, 8, 9, 20, 25], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7, 9, 10, 21, 27], dtype=np.float64)


# creating the data set to check if the algorithm is working correctly
def create_dataset(hm, variance, step=2, correlation=False):
    # creating psudo random values for our dataset
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step
        xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()


# linear regression y = mx + y intercept
# where m is the best fit slope function
# defining the best fit slope function
# takes data as argument
def best_fit_slope_and_intercept(xs, ys):
    # equation is mean of the x multiplied mean of the y - mean of x times y
    # divided by the mean of the x * mean of the x - mean of x squared
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs) * mean(xs)) - mean(xs*xs)) )
    # equation for the y intercept
    # y intercept(b) = mean of y - m times mean of x
    b = mean(ys) - m * mean(xs)
    return m, b


# creating a function that checks how good our best fit line is
# that is creating a function for r squared error
# r squared error is the distance of the data points to the line squared
def squared_error(ys_orig, ys_line):

    return sum((ys_line - ys_orig)**2)


# creating a function that claculates the coefficient of the determination
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


xs, ys = create_dataset(40, 10, 2, correlation="pos")

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)
# pemdas = parenthesis off, exponential operation first then multiplication  rest we know
# making a regression line
regression_line = [(m*x) + b for x in xs]

# predicting for data point 15
predict_x = 15
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
# plotting fot the data
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()
