import pandas as pd

from src.multi_mse_visual import plot_multi_mse


import matplotlib.pyplot as plt
from pandas.plotting import table 
from matplotlib.font_manager import FontProperties

def multi_mse(mse_fname, show_group=False):
    df = pd.read_csv(mse_fname, index_col='model')

    df = df.round(3)
    
    col_list = list(df)
    min_mse = df[col_list].min(axis=1).min()
    max_mse = df[col_list].max(axis=1).max()
    winner_idx = df[col_list].min(axis=1).idxmin()
    no_of_bins = max_mse // 50

    print("%s %s (%s) \n" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin()))
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

    #print("Min: %s, Max: %s" % (min_mse, max_mse))

    plot_multi_mse(df, min_mse, no_of_bins)
    df_arr = []

    if show_group:
        df_arr.append(df.iloc[:15,:])
        df_arr.append(df.iloc[15:27,:])
        df_arr.append(df.iloc[27:,:])

    for i in range(len(df_arr)):
        df = df_arr[i]
        col_list = list(df)
        min_mse = df[col_list].min(axis=1).min()
        max_mse = df[col_list].max(axis=1).max()
        no_of_bins = max_mse // 50

        display(df)
        print("Min: %s, Max: %s" % (min_mse, max_mse))
        plot_multi_mse(df, min_mse, no_of_bins)