import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self) -> None:
        self.desc_fixed_len = 200
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # Z score scalers:
        self.runtime_scaler = StandardScaler()
        self.budget_scaler = StandardScaler()
        self.vote_avg_scaler = StandardScaler()
        self.vote_count_scaler = StandardScaler()

    def print_columns_names(self):
        print(self.df.columns)

    def remove_by_lower_bound(self, column_name, lower_bound=0):
        self.df = self.df[self.df[column_name] > lower_bound]

    def remove_low_revenue_movies(self):
        self.remove_by_lower_bound("revenue", 500_000)

    def remove_low_budget_movies(self):
        self.remove_by_lower_bound("budget", 10_000)

    def remove_unused_features(self, features_str_lst):
        self.df = self.df.drop(columns=features_str_lst)

    def create_description_column(self):
        """
        Combine title and overview into one column, description.
        """
        self.df["description"] = self.df["title"] + self.df["overview"]
        self.df = self.df.drop(columns=["title", "overview"])

    def split_genres_column(self):
        """
        Apply one-hot encoding to the genres column.
        That is, for each genre, create a binary feature.
        """
        one_hot_genres = self.df["genres"].str.get_dummies(sep="-")
        self.df = self.df.drop(columns=["genres"])
        self.df = pd.concat([self.df, one_hot_genres], axis=1)

    def remove_empty_description(self):
        self.df = self.df.dropna(subset=["description"])

    def replace_zero_with_average(self, column_str_lst):
        """
        This function replaces all zero values in a given columns with
        the average of those columns.
        """
        for column_name in column_str_lst:
            average_value = self.df[column_name].mean()
            self.df.loc[self.df[column_name] == 0, column_name] = average_value

    def tokenize_description(self):
        desc_tokenizer = lambda desc: self.tokenizer(
            desc,
            padding="max_length",
            truncation=True,
            max_length=self.desc_fixed_len,
            return_tensors="pt",
        )["input_ids"].squeeze(dim=0)
        input_ids = self.df["description"].apply(desc_tokenizer)
        self.df["description"] = input_ids

    def normalize_column(self, normalize_amount, column_name):
        self.df[column_name] = self.df[column_name].apply(
            lambda revenue: revenue / normalize_amount
        )

    def normalize_revenue(self):
        self.normalize_column(1_000_000, "revenue")

    def normalize_budget(self):
        self.normalize_column(1_000_000, "budget")

    def fit_scalers_with_df(self):
        self.runtime_scaler.fit(self.df[["runtime"]])
        self.budget_scaler.fit(self.df[["budget"]])
        self.vote_avg_scaler.fit(self.df[["vote_average"]])
        self.vote_count_scaler.fit(self.df[["vote_count"]])

    def transform_df_with_scalers(self):
        self.df["runtime"] = self.runtime_scaler.transform(self.df[["runtime"]])
        self.df["budget"] = self.budget_scaler.transform(self.df[["budget"]])
        self.df["vote_average"] = self.vote_avg_scaler.transform(self.df[["vote_average"]])
        self.df["vote_count"] = self.vote_count_scaler.transform(self.df[["vote_count"]])


    def preprocess_for_train(self, train_df):
        self.df = train_df
        self.remove_low_revenue_movies()
        self.remove_low_budget_movies()
        unused_features = [
            "original_language",
            "release_date",
            "popularity",
            "status",
            "tagline",
            "keywords",
            "poster_path",
            "backdrop_path",
            "recommendations",
            "id",
            "production_companies",
            "credits",
        ]
        self.remove_unused_features(unused_features)
        self.create_description_column()
        self.split_genres_column()
        self.remove_empty_description()
        zero_with_average_lst = ["budget", "runtime", "vote_average", "vote_count"]
        self.replace_zero_with_average(zero_with_average_lst)
        self.normalize_revenue()
        self.normalize_budget()
        self.tokenize_description()
        self.fit_scalers_with_df()
        self.transform_df_with_scalers()
        return self.df

    def preprocess_for_predict(self, single_movie_df):
        self.df = single_movie_df
        self.create_description_column()
        genre_list = [
                "Action",
                "Adventure",
                "Animation",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Family",
                "Fantasy",
                "History",
                "Horror",
                "Music",
                "Mystery",
                "Romance",
                "Science Fiction",
                "TV Movie",
                "Thriller",
                "War",
                "Western",
            ]
        cur_genre = self.df["genres"]
        for genre in genre_list:
            self.df[genre] = 0
        self.df[cur_genre] = 1
        self.df = self.df.drop(columns=["genres"])
        self.tokenize_description()
        self.transform_df_with_scalers()
        return self.df

def main():
    dict = {"title":"Avi", "genres":"Thriller", "overview":"I love pizza"}
    preprocessor = Preprocessor(pd.DataFrame.from_dict(dict), 200)
    df = preprocessor.preprocess_single_movie()


if __name__ == "__main__":
    main()
