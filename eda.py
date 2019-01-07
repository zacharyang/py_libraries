### Functions for Online News Popularity Analysis ### 

def tab_missing_values(df):
    """
        Takes a data frame and tabulates all missing values by column 
    """    
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']

    # Filling factor is % of non null values in the column #
    missing_df['filling_factor'] = (df.shape[0] 
                                    - missing_df['missing_count']) / df.shape[0] * 100
    missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)

    return missing_df 


def get_redundant_pairs(df):
    '''
        Get diagonal and lower triangular pairs of correlation matrix
    '''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df):
    """
        Get top correlated variables by absolute correlation
    """

    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr


def tri_corr_heatmap(df):
    """
        Get top correlated variables by absolute correlation
    """

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df.corr(),mask=mask,annot=True, cmap='RdBu')

    