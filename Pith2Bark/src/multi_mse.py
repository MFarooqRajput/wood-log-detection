import pandas as pd

from src.multi_mse_visual import plot_multi_mse

def multi_mse(mse_fname, show_group=True):
    df = pd.read_csv(mse_fname, index_col='model')
    col_list = list(df)
    min_mse = df[col_list].min(axis=1).min()
    max_mse = df[col_list].max(axis=1).max()
    winner_idx = df[col_list].min(axis=1).idxmin()
    no_of_bins = max_mse // 50

    print("%s %s (%s) \n" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin()))
    display(df)
    print("Min: %s, Max: %s" % (min_mse, max_mse))

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