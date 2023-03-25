import pandas as pd
import numpy as np

EXCEL_EXTENSIONS = ['.xsl', '.xlsx']

def is_excel_file(filename):
    return any(filename.endswith(extension) for extension in EXCEL_EXTENSIONS)

def read_rings_ranking(fname):
    try:
        if is_excel_file(fname):
            
            # read excel file
            ringsranking = pd.read_excel(fname)

            # fill nan
            ringsranking = ringsranking.fillna('0 st')

            # make df
            ringsranking_df = pd.DataFrame(ringsranking)

            # rename columns
            ringsranking_df.rename(columns={'Bildlänk': 'image_name', 'Räkning 1': 'ranking_1', 'Räkning 2': 'ranking_2', 'Räkning 3': 'ranking_3'}, inplace=True)
            
            # convert to float
            ringsranking_df['ranking_1'] = ringsranking_df['ranking_1'].str.replace(r'\D+', '', regex=True).astype(float)
            ringsranking_df['ranking_2'] = ringsranking_df['ranking_2'].str.replace(r'\D+', '', regex=True).astype(float)
            ringsranking_df['ranking_3'] = ringsranking_df['ranking_3'].str.replace(r'\D+', '', regex=True).astype(float)

            # replace 0 with na
            ringsranking_df.replace(0, np.nan, inplace=True)

            # fill nan with mean
            ringsranking_df = ringsranking_df.T.fillna(ringsranking_df[['ranking_1', 'ranking_2', 'ranking_3']].mean(axis=1)).T

            # convert to int
            ringsranking_df['ranking_1'] = ringsranking_df['ranking_1'].astype(int)
            ringsranking_df['ranking_2'] = ringsranking_df['ranking_2'].astype(int)
            ringsranking_df['ranking_3'] = ringsranking_df['ranking_3'].astype(int)

            ringsranking_df['ranking'] = ringsranking_df.mean(numeric_only=True, axis=1)

            return ringsranking_df
            
    except Exception as e:
        print(e)

def read_rings_count(fname):
    try:
        if is_excel_file(fname):
            
            # read excel file
            ringsranking = pd.read_excel(fname)

            # fill nan
            ringsranking = ringsranking.fillna('0')

            # make df
            ringsranking_df = pd.DataFrame(ringsranking)

            # rename columns
            ringsranking_df.rename(columns={'Image': 'image_name', 'Count 1': 'ranking_1', 'Count 2': 'ranking_2', 'Count 3': 'ranking_3', 'Count 4': 'ranking_4', 'Count 5': 'ranking_5'}, inplace=True)
            
            # convert to float
            ringsranking_df['ranking_1'] = ringsranking_df['ranking_1'].astype(float)
            ringsranking_df['ranking_2'] = ringsranking_df['ranking_2'].astype(float)
            ringsranking_df['ranking_3'] = ringsranking_df['ranking_3'].astype(float)
            ringsranking_df['ranking_4'] = ringsranking_df['ranking_4'].astype(float)
            ringsranking_df['ranking_5'] = ringsranking_df['ranking_5'].astype(float)

            # replace 0 with na
            ringsranking_df.replace(0, np.nan, inplace=True)

            # fill nan with mean
            ringsranking_df = ringsranking_df.T.fillna(ringsranking_df[['ranking_1', 'ranking_2', 'ranking_3', 'ranking_4', 'ranking_5']].mean(axis=1)).T

            # convert to int
            ringsranking_df['ranking_1'] = ringsranking_df['ranking_1'].astype(int)
            ringsranking_df['ranking_2'] = ringsranking_df['ranking_2'].astype(int)
            ringsranking_df['ranking_3'] = ringsranking_df['ranking_3'].astype(int)
            ringsranking_df['ranking_4'] = ringsranking_df['ranking_4'].astype(int)
            ringsranking_df['ranking_5'] = ringsranking_df['ranking_5'].astype(int)

            ringsranking_df['ranking'] = ringsranking_df.mean(numeric_only=True, axis=1)
            
            return ringsranking_df
            
    except Exception as e:
        print(e)