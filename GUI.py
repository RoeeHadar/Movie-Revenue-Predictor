import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from tkinter import messagebox

from revenue_prediction_model import MyModel

class MovieRevenuePredictorGUI:
    def __init__(self, master):

        # Predictor related init:
        MODEL_SAVE_PATH = "./data"
        self.model = MyModel(has_dynamic_lr=True)
        self.model.load_params_from_file(MODEL_SAVE_PATH)

        # GUI related init:
        self.master = master
        self.master.geometry("350x300")
        self.master.title("Movie Revenue Predictor")
        self.defaultFont = tkFont.nametofont("TkDefaultFont")
        self.defaultFont.configure(family="Calibri", size=16)

        # Create labels and entry widgets for movie features
        feature_counter = 0

        tk.Label(master, text="Movie Title:").grid(row=feature_counter, column=0)
        self.title_entry = tk.Entry(master)
        self.title_entry.grid(row=feature_counter, column=1)
        feature_counter += 1

        tk.Label(master, text="Movie Overview:").grid(row=feature_counter, column=0)
        self.overview_entry = tk.Entry(master)
        self.overview_entry.grid(row=feature_counter, column=1)
        feature_counter += 1

        tk.Label(master, text="Budget ($ million):").grid(row=feature_counter, column=0)
        self.budget_entry = tk.Entry(master)
        self.budget_entry.grid(row=feature_counter, column=1)
        feature_counter += 1

        tk.Label(master, text="Runtime (minutes):").grid(row=feature_counter, column=0)
        self.runtime_entry = tk.Entry(master)
        self.runtime_entry.grid(row=feature_counter, column=1)
        feature_counter += 1

        tk.Label(master, text="Vote Average (1-10):").grid(
            row=feature_counter, column=0
        )
        self.vote_avg_entry = tk.Entry(master)
        self.vote_avg_entry.grid(row=feature_counter, column=1)
        feature_counter += 1

        tk.Label(master, text="Number of Voters:").grid(row=feature_counter, column=0)
        self.num_of_voters_entry = tk.Entry(master)
        self.num_of_voters_entry.grid(row=feature_counter, column=1)
        feature_counter += 1

        # Dropdown menu for genre
        tk.Label(master, text="Genre:").grid(row=feature_counter, column=0)
        self.genre_var = tk.StringVar()
        self.genre_combobox = ttk.Combobox(
            master,
            textvariable=self.genre_var,
            values=[
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
            ],
        )
        self.genre_combobox.grid(row=feature_counter, column=1)
        feature_counter += 1

        # Create a button to predict revenue
        tk.Button(master, text="Predict Revenue", command=self.predict_revenue).grid(
            row=feature_counter, column=0, columnspan=2, pady=10
        )

    def predict_revenue(self):
        # Get user input
        title = self.title_entry.get()
        overview = self.overview_entry.get()
        budget_str = self.budget_entry.get()
        budget = float(budget_str) if budget_str != "" else 0
        runtime_str = self.runtime_entry.get()
        runtime = int(runtime_str) if runtime_str != "" else 0
        vote_avg_str = self.vote_avg_entry.get()
        vote_avg = float(vote_avg_str) if vote_avg_str != "" else 0
        vote_counts_str = self.num_of_voters_entry.get()
        vote_counts = int(vote_counts_str) if vote_counts_str != "" else 0
        genre = self.genre_var.get()

        # Use your model to make a prediction
        prediction = self.model.predict(movie_title=title,
                       movie_overview=overview,
                       budget_in_millions=budget,
                       runtime_in_minutes=runtime,
                       vote_avg=vote_avg,
                       number_of_voters=vote_counts,
                       genre=genre)

        # Display the prediction in a messagebox
        messagebox.showinfo(
            "Revenue Prediction", f"The predicted revenue is: ${prediction:.2f} million"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRevenuePredictorGUI(root)
    root.mainloop()
