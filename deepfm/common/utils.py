import polars as pl


def make_feature():
    movies_df = pl.read_csv("movies.csv")
    ratings_df = pl.read_csv("ratings.csv")
    tags_df = pl.read_csv("tags.csv")

    concat_df = pl.concat([movies_df, ratings_df, tags_df], how="align")
    concat_df = concat_df.unique(maintain_order=True).to_pandas()
    concat_df.dropna(axis=0, inplace=True)
    concat_df = concat_df[["userId", "movieId", "rating", "genres"]]

    all_genres = set()
    for genres in concat_df["genres"]:
        all_genres.update(genres.split("|"))
    all_genres = sorted(list(all_genres))

    # 각 장르에 대한 새로운 열 생성 및 라벨링
    for genre in all_genres:
        concat_df[genre] = concat_df["genres"].apply(lambda x: 1 if genre in x else 0)
    concat_df["rating"] = concat_df["rating"].apply(lambda x: 0 if x <= 3 else 1)
    concat_df["userId"] = concat_df["userId"].astype(int)
    concat_df = concat_df.drop("genres", axis=1)

    feature_columns = concat_df.columns.to_list()
    feature_columns.remove("rating")

    return concat_df, feature_columns
