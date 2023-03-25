import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def heat_map(df, max=1, min=-1):

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df, dtype=bool),1)

    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(11, 9))
    f, ax = plt.subplots(figsize=(4, 3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    labels = ['Not Damage', 'Damaged']
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, mask=mask, cmap=cmap, vmax=max, center=(max+min)/2, vmin=min,
                square=True, annot=True, linewidths=.2, cbar_kws={"shrink": .5},
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.show()

def plot_accuracy(x, y, tick_label):
    plt.bar(x, y, tick_label=tick_label, width=0.8)
    
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model vs Accuracy')
    plt.show()