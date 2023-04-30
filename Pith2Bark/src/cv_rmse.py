import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table 
from matplotlib.font_manager import FontProperties

def plot_multi_rmse(df, min_rmse, no_of_bins):
    df.plot(kind="bar", figsize=(12, 4))
    
    plt.title("Model vs RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.axhline(y=min_rmse, color='g', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.locator_params(axis='y', nbins=no_of_bins)
    plt.show()

def multi_rmse(rmse_fname, show_group=False):
    df = pd.read_csv(rmse_fname, index_col='model')

    #df = df.round(3)
    
    col_list = list(df)
    min_rmse = df[col_list].min(axis=1).min()
    max_rmse = df[col_list].max(axis=1).max()
    winner_idx = df[col_list].min(axis=1).idxmin()
    no_of_bins = max_rmse // 50

    print("%s %s (%s) \n" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin()))
    #print("Min: %s, Max: %s" % (min_rmse, max_rmse))
    #display(df)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('off')
    table =  ax.table(cellText=df.values, colLabels=['SubSeq', 'Peaks', 'Binary', 'MWA'], loc='center', rowLabels=df.index)

    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text='Model', loc='center')

    for (row, col), cell in table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    
    plt.show()

    plot_multi_rmse(df, min_rmse, no_of_bins)
    df_arr = []

    if show_group:
        df_arr.append(df.iloc[:15,:])
        df_arr.append(df.iloc[15:27,:])
        df_arr.append(df.iloc[27:,:])

    for i in range(len(df_arr)):
        df = df_arr[i]
        col_list = list(df)
        min_rmse = df[col_list].min(axis=1).min()
        max_rmse = df[col_list].max(axis=1).max()
        no_of_bins = max_rmse // 50

        display(df)
        print("Min: %s, Max: %s" % (min_rmse, max_rmse))
        plot_multi_rmse(df, min_rmse, no_of_bins)