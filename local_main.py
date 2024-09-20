# Imports
import pandas as pd
from revenue_prediction_model import BertBasedModel
from create_data_loader import DataloaderMaker

# Constants
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS_AMOUNT = 3
TEST_PERCENTAGE = 0.25
CSV_FILE_PATH = "G:\Technion\Courses\Semester 9\Project AI - 236502\code\data\movies.csv"

# Loading data
initial_data_frame = pd.read_csv(CSV_FILE_PATH)
my_model = BertBasedModel(has_dynamic_lr=True)

# Preprocess data
processed_df = my_model.preprocess_dataframe(initial_data_frame)
print(f"features names: {processed_df.columns}")
print(f"number of samples before pre-processing: {len(initial_data_frame)}")
print(f"number of samples after pre-processing: {len(processed_df)}")

# Create data loader
data_loader_maker = DataloaderMaker(processed_df,
                                    batch_size=BATCH_SIZE,
                                    test_percentage=TEST_PERCENTAGE)
train_data_loader = data_loader_maker.get_train_iterator()
test_data_loader = data_loader_maker.get_test_iterator()

# Train the model
my_model.train_all(train_data_loader, EPOCHS_AMOUNT, LEARNING_RATE)

# Test the model
test_loss = my_model.evaluate(test_data_loader)
print(f"Test loss: {test_loss}")