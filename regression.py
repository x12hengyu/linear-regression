import random
import numpy as np
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = np.genfromtxt(filename, delimiter=',')
    dataset = dataset[1::, :][:, 1::]
    return dataset


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    c = dataset[:, col]
    print(len(c))
    print(np.around(np.mean(c), decimals=2))
    print(np.around(np.std(c), decimals=2))


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    for row in dataset:
        mse += (betas[0] + np.dot(row[cols], betas[1:]) - row[0]) ** 2
    return mse/len(dataset)


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    for i in range(len(betas)):
        total = 0
        for row in dataset:
            grad = (betas[0] + np.dot(row[cols], betas[1:]) - row[0])    
            if i != 0:
                total += grad * row[cols[i-1]]
            else: 
                total += grad
        grads.append(total * 2 / len(dataset))
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    for i in range(1, T+1):
        grads = gradient_descent(dataset, cols, betas)
        for j in range(len(betas)):
            betas[j] = betas[j] - eta * grads[j]
        mse = regression(dataset, cols, betas)
        b = ['{:.2f}'.format(n) for n in betas]
        print(i, '{:.2f}'.format(mse), *b)



def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    dataset = np.array(dataset)
    X = dataset[:, cols]
    X = np.insert(X, 0, 1, axis=1)
    y = dataset[:, 0]
    betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = np.array(compute_betas(dataset, cols)[1:])
    features = np.append([1], features)
    result = np.dot(features, betas)
    return result

def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear = []
    quadra = []
    for xi in X:
        y_linear = betas[0] + betas[1] * xi[0] + np.random.normal(scale = sigma)
        linear.append([y_linear, xi[0]])

        y_quadra = alphas[0] + alphas[1] * xi[0] ** 2 + np.random.normal(scale = sigma)
        quadra.append([y_quadra, xi[0]])

    return np.array(linear), np.array(quadra)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph

    X = []
    for i in range(1000):
        X.append([random.randint(-100, 100)])

    betas, alphas = [1, 2], [2, 4]

    sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    syn = []
    for s in sigmas:
        syn.append(synthetic_datasets(betas, alphas, X, s))

    l_mse, q_mse = [], []
    for i in range(len(syn)):
        l_mse.append(compute_betas(syn[i][0], cols=[1])[0])
        q_mse.append(compute_betas(syn[i][1], cols=[1])[0])

    plt.plot(sigmas, l_mse, "-o")
    plt.plot(sigmas, q_mse, "-o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("different settings of sigma")
    plt.ylabel("MSEs calculated from the linear and quadratic datasets")
    plt.legend(["linear dataset", "quadratic dataset"])
    plt.savefig("mse.pdf", format="pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
