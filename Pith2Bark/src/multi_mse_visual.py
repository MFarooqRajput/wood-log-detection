import matplotlib.pyplot as plt

def plot_multi_mse(df, min_mse, no_of_bins):
    df.plot(kind="bar", figsize=(12, 4))
    
    plt.title("Model vs RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.axhline(y=min_mse, color='g', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.locator_params(axis='y', nbins=no_of_bins)
    plt.show()