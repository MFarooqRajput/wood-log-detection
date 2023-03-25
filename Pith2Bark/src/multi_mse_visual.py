import matplotlib.pyplot as plt

def plot_multi_mse(df, min_mse, no_of_bins):
    df.plot(kind="bar")
    
    plt.title("Model vs MSE")
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.axhline(y=min_mse, color='g', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.locator_params(axis='y', nbins=no_of_bins)
    plt.show()