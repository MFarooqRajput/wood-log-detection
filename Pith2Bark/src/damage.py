import pandas as pd
import numpy as np

def damage(df):

    df['diff_min_max'] = df['max'] - df['min']
    df['diff_orig_mean'] = abs(df['orig'] - df['mean'])
    
    df['damage_1'] = np.where(df['diff_min_max'] > 39, 1, 0)
    df['damage_2'] = np.where(df['diff_orig_mean'] > 14, 1, 0)
    
    return df[['image', 'min', 'max', 'mean', 'orig', 'diff_min_max', 'diff_orig_mean', 'damage_1', 'damage_2']]

def merge_df(df_1, df_2, df_3, df_4, df_5):
    df = pd.concat([df_1, df_2, df_3, df_4, df_5])
    return df

def save_damage(damage_fname, df):
    with open(damage_fname, 'a') as f:
        df.to_csv(f, header=f.tell()==0, index = False)

    return df






def read_damage(damage_fname, df_1, df_2, df_3, df_4, df_5):
    df_damage = pd.read_csv(damage_fname)
    
    df_4_sel = df_4[(df_4['image'] == '6693.jpg')]
    df_4_min = df_4_sel['min'].values[0]
    df_4_max = df_4_sel['max'].values[0]
    df_4 = df_4[(df_4['image'] != '6693.jpg')]

    df_damage_ind = df_damage[(df_damage['image'] == '6693.jpg') & (df_damage['min'] == df_4_min) & (df_damage['max'] == df_4_max)].index
    df_damage.drop(df_damage_ind, inplace = True )

    columns = ['min', 'max', 'mean', 'orig', 'diff_min_max', 'diff_orig_mean', 'damage_2']
    df_damage_temp = df_damage.drop(columns=columns)
    df_damage_temp.rename(columns={'damage_1': 'Damage'}, inplace=True)

    df_concat = pd.concat([df_1, df_2, df_3, df_4, df_5])
    df = pd.merge(left=df_damage_temp, right=df_concat, left_on="image", right_on="image")
    df.reset_index(drop=True, inplace=True)

    return df


def read_damage(damage_fname, df_concat):
    df_damage = pd.read_csv(damage_fname)
    
    columns = ['min', 'max', 'mean', 'orig', 'diff_min_max', 'diff_orig_mean', 'damage_2']
    df_damage_temp = df_damage.drop(columns=columns)
    df_damage_temp.rename(columns={'damage_1': 'Damage'}, inplace=True)

    df = pd.merge(left=df_damage_temp, right=df_concat, left_on="image", right_on="image")
    df.reset_index(drop=True, inplace=True)

    return df