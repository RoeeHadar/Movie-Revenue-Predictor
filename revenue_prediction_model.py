from transformers import DistilBertModel
import torch
from torch.optim import Adam
import torch.nn as nn
from preprocess import Preprocessor
import pandas as pd
import os
import pickle
from tqdm import tqdm


class BaseModel(nn.Module):
    """
    Movie's revenue prediction model.
    Our model is utilaizing an existing NLP model for parsing of movie description,
    and a few fully connected layers for the non-textual features and description output.
    """
    def __init__(self, has_dynamic_lr=False, nlp_out_dim=768):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features_count = 23
        output_dim = 1
        cur_hidden_dim = features_count + nlp_out_dim
        self.hidden_dim1 = cur_hidden_dim
        self.hidden_dim2 = cur_hidden_dim
        self.hidden_dim3 = cur_hidden_dim
        self.sigma = torch.nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_dim1, self.hidden_dim2, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim2, self.hidden_dim3, bias=True)
        self.fc3 = nn.Linear(self.hidden_dim3, output_dim, bias=True)
        self.training_loss_per_epoch = []
        self.has_dynamic_lr = has_dynamic_lr
        self.to(self.device)

    def train_all(self, train_iter, num_of_epochs, learning_rate):
        """
        The training loop.
        * compute predictions with forward pass.
        * compute gradients with backward pass.
        * update weights using the gradients.
        """
        self.train()
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(num_of_epochs)):
            total_loss = 0
            average_loss_per_movie = 0
            total_number_of_movies = 0
            for batch in train_iter:
                desc_batch, other_features_batch, revenues_batch = batch
                # Adding current batch size to total.
                total_number_of_movies += revenues_batch.size()[0]
                optimizer.zero_grad()
                revenues_predictions = self.forward(desc_batch, other_features_batch)
                loss = criterion(revenues_predictions, revenues_batch)
                if loss.isnan():
                    continue
                # Compute gradients.
                loss.backward()
                # Update the weights.
                optimizer.step()
                current_loss = loss.item()
                total_loss += current_loss
            average_loss_per_movie = total_loss / total_number_of_movies
            print(
                f"Epoch {epoch+1}: Average loss in $millions$ for a movie: {average_loss_per_movie}"
            )
            self.training_loss_per_epoch.append(int(average_loss_per_movie))
            # Dynamic learning rate update:
            if self.has_dynamic_lr:
                # 3.167 is square-root of 10
                learning_rate = learning_rate / 3.167
                for g in optimizer.param_groups:
                    g["lr"] = learning_rate

    def evaluate(self, test_iter):
        """
        Evaluate the model's performance on the test data.
        Returns the average loss on the test iterator.
        """
        self.eval()
        criterion = nn.MSELoss()
        total_number_of_test_movies = 0
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_iter):
                desc_batch, other_features_batch, revenues_batch = batch
                # Adding current batch size to total.
                total_number_of_test_movies += revenues_batch.size()[0]
                revenues_predictions = self.forward(desc_batch, other_features_batch)
                loss = criterion(revenues_predictions, revenues_batch)
                total_loss += loss.item()
        return total_loss / total_number_of_test_movies

    def preprocess_dataframe(self, df):
        return self.preprocessor.preprocess_for_train(df)

class BertBasedModel(BaseModel):
    def __init__(self, has_dynamic_lr=False, nlp_out_dim=768):
        super().__init__(has_dynamic_lr, nlp_out_dim)
        self.nlp_model =  DistilBertModel.from_pretrained("distilbert-base-uncased").to(
            self.device
        )

    def save_params_to_file(self, model_save_dir_path: str):
        model_file_name = "model.pth"
        preprocessor_file_name = "preprocessor.pkl"
        model_save_path = os.path.join(model_save_dir_path, model_file_name)
        preprocessor_save_path = os.path.join(
            model_save_dir_path, preprocessor_file_name
        )
        torch.save(self.state_dict(), model_save_path)
        with open(preprocessor_save_path, "wb") as f:
            pickle.dump(self.preprocessor, f)

    def load_params_from_file(self, model_save_dir_path: str):
        model_file_name = "model.pth"
        preprocessor_file_name = "preprocessor.pkl"
        model_save_path = os.path.join(model_save_dir_path, model_file_name)
        preprocessor_save_path = os.path.join(
            model_save_dir_path, preprocessor_file_name
        )
        self.load_state_dict(torch.load(model_save_path, map_location=self.device))
        with open(preprocessor_save_path, "rb") as f:
            self.preprocessor = pickle.load(f)

    def forward(self, desc_batch, other_features_batch):
        """
        compute predictions - returns a tensor of the revenues for each movie in the batch.
        other_features_batch size - batch_size x 23
        desc_batch size - batch_size x 200
        """
        self.nlp_model.eval()
        with torch.no_grad():
            distilbert_out = self.nlp_model(desc_batch)
        # last_hidden_state dim is [batch_size, 200, 768]
        last_hidden_state = distilbert_out.last_hidden_state
        # Mean pooling on the tokens dim.
        mean_pooled_distilbert_out = last_hidden_state.mean(dim=1)
        concatenated_tensor = torch.cat(
            (mean_pooled_distilbert_out, other_features_batch), dim=1
        )
        fc1_output = self.fc1(concatenated_tensor)
        fc1_output = self.sigma(fc1_output)
        fc2_output = self.fc2(fc1_output)
        fc2_output = self.sigma(fc2_output)
        fc3_output = self.fc3(fc2_output)
        return fc3_output.squeeze(1)

    def predict(
        self,
        movie_title: str,
        movie_overview: str,
        budget_in_millions: float,
        runtime_in_minutes: int,
        vote_avg: float,
        number_of_voters: int,
        genre: str,
    ):
        self.eval()

        movie_dict = {
            "title": [movie_title],
            "overview": [movie_overview],
            "budget": [budget_in_millions],
            "runtime": [runtime_in_minutes],
            "vote_average": [vote_avg],
            "vote_count": [number_of_voters],
            "genres": [genre],
        }

        single_movie_df = pd.DataFrame.from_dict(movie_dict)
        single_movie_df = self.preprocessor.preprocess_for_predict(
            single_movie_df
        ).iloc[0]
        description = single_movie_df["description"]
        features = single_movie_df.drop("description")
        tensor_description = description.to(self.device)
        tensor_features = torch.tensor(
            features, dtype=torch.float32, device=self.device
        )
        tensor_description = tensor_description.unsqueeze(dim=0)
        tensor_features = tensor_features.unsqueeze(dim=0)
        revenue_prediction = self.forward(tensor_description, tensor_features)
        return revenue_prediction.item()

    def preprocess_dataframe(self, df):
        return self.preprocessor.preprocess_for_train(df)

class RnnBasedModel(BaseModel):
    def __init__(self, has_dynamic_lr=False, nlp_out_dim=768):
        super().__init__(has_dynamic_lr, nlp_out_dim)
        self.nlp_model = nn.LSTM(input_size=200, hidden_size=768, num_layers=1, batch_first=True).to(
            self.device
        )

    def forward(self, desc_batch, other_features_batch):
        """
        compute predictions - returns a tensor of the revenues for each movie in the batch.
        other_features_batch size - batch_size x 23
        desc_batch size - batch_size x 200
        """
        self.nlp_model.eval()
        with torch.no_grad():
            lstm_out, _ = self.nlp_model(desc_batch)
        concatenated_tensor = torch.cat((lstm_out, other_features_batch), dim=1)
        fc1_output = self.fc1(concatenated_tensor)
        fc1_output = self.sigma(fc1_output)
        fc2_output = self.fc2(fc1_output)
        fc2_output = self.sigma(fc2_output)
        fc3_output = self.fc3(fc2_output)
        return fc3_output.squeeze(1)
