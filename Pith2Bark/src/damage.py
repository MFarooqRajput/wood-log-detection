import pandas as pd
import numpy as np

def damage(df):

    df['diff_min_max'] = df['max'] - df['min']
    df['diff_orig_mean'] = abs(df['orig'] - df['mean'])
    
    df['damage_min_max'] = np.where(df['diff_min_max'] > 39, 1, 0)
    df['damage_mean'] = np.where(df['diff_orig_mean'] > 14, 1, 0)
    
    return df[['image', 'min', 'max', 'mean', 'orig', 'diff_min_max', 'diff_orig_mean', 'damage_min_max', 'damage_mean']]

def merge_df(df_1, df_2, df_3, df_4, df_5):
    df = pd.concat([df_1, df_2, df_3, df_4, df_5])
    return df

def save_damage(damage_fname, df):
    with open(damage_fname, 'a') as f:
        df.to_csv(f, header=f.tell()==0, index = False)

    return df

def read_damage_min_max(damage_fname, df_concat):
    df_damage = pd.read_csv(damage_fname)
    
    columns = ['min', 'max', 'mean', 'orig', 'diff_min_max', 'diff_orig_mean', 'damage_mean']
    df_damage_temp = df_damage.drop(columns=columns)
    df_damage_temp.rename(columns={'damage_min_max': 'Damage'}, inplace=True)

    df = pd.merge(left=df_damage_temp, right=df_concat, left_on="image", right_on="image")
    df.reset_index(drop=True, inplace=True)

    return df

def read_damage_mean(damage_fname, df_concat):
    df_damage = pd.read_csv(damage_fname)
    
    columns = ['min', 'max', 'mean', 'orig', 'diff_min_max', 'diff_orig_mean', 'damage_min_max']
    df_damage_temp = df_damage.drop(columns=columns)
    df_damage_temp.rename(columns={'damage_mean': 'Damage'}, inplace=True)

    df = pd.merge(left=df_damage_temp, right=df_concat, left_on="image", right_on="image")
    df.reset_index(drop=True, inplace=True)

    return df