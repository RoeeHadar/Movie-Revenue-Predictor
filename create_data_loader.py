from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split


class MovieDataset(Dataset):
    def __init__(self, movies_features, movies_revenues):
        self.X = movies_features
        self.y = movies_revenues
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X.iloc[index]
        label = self.y.iloc[index]
        description = features["description"]
        features = features.drop("description")
        tensor_description = description.to(self.device)
        tensor_features = torch.tensor(
            features, dtype=torch.float32, device=self.device
        )
        tensor_revenue_label = torch.tensor(
            label, dtype=torch.float32, device=self.device
        )
        return tensor_description, tensor_features, tensor_revenue_label


class DataloaderMaker:
    def __init__(self, df, batch_size=64, test_percentage=0.25) -> None:
        self.df = df.copy()
        X = self.df.loc[:, self.df.columns != "revenue"]
        y = self.df["revenue"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_percentage
        )
        train_dataset = MovieDataset(X_train, y_train)
        self.train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        test_dataset = MovieDataset(X_test, y_test)
        self.test_data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    def get_train_iterator(self):
        return self.train_data_loader

    def get_test_iterator(self):
        return self.test_data_loader
