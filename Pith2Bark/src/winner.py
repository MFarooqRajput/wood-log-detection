import pandas as pd
import numpy as np

from src.visual import plot_rings_along_lines
from src.visual import plot_min_max_mean_orig

def winner_model_algo(mse_fname):
    df = pd.read_csv(mse_fname, index_col='model')
    col_list = list(df)
    winner_idx = df[col_list].min(axis=1).idxmin()

    print("%s %s (%s)" % (winner_idx, df.loc[winner_idx].min(), df.loc[winner_idx].idxmin())) 
    #print(df.loc[winner_idx])
    
    return winner_idx, df.loc[winner_idx].idxmin()

def winner_rings(rings_fname, winner_idx, algo):
    df = pd.read_csv(rings_fname, index_col='model') 
    sub_df = df.loc[winner_idx:winner_idx]
    rings_df = sub_df.loc[sub_df['algo'] == algo]
    #display(rings_df)
    return rings_df

def winner_model_rings(mse_fname, rings_fname):
    model, algo = winner_model_algo(mse_fname)
    df = winner_rings(rings_fname, model, algo)

    return df

def winner_visual(df):
    lines_cols = [col for col in df.columns if 'line' in col]
    lines = [s.strip('line_') for s in lines_cols]
    min_max_mean = []
    algo_name = None
    no_of_images = 0

    for index, row in df.iterrows():
        print("%s %s %s" % (row['image'], row['mean'], row['orig']))
        name = row['algo']
        x = lines
        y = row[lines_cols].values
        ranking = row['orig']
        plot_rings_along_lines(name, x, y, ranking)

        min_max_mean.append([row['min'], row['max'], row['mean'], row['orig']])

        if algo_name is None:
            algo_name = name

        no_of_images += 1

    x = np.arange(start=1, stop=no_of_images + 1, step=1)
    y = np.array(min_max_mean)
    plot_min_max_mean_orig(algo_name, x, y)