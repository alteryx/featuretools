def get_df_tags(df):
    """Gets a DataFrame's semantic tags without index or time index tags for Woodwork init"""
    semantic_tags = {}
    for col_name in df.columns:
        semantic_tags[col_name] = df.ww.semantic_tags[col_name] - {
            "time_index",
            "index",
        }

    return semantic_tags
