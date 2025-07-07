import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample F1-style dataset
data = pd.DataFrame({
    "driver": ["Verstappen", "Leclerc", "Hamilton", "Norris", "Sainz"] * 20,
    "team": ["Red Bull", "Ferrari", "Mercedes", "McLaren", "Ferrari"] * 20,
    "qual_pos": list(range(1, 6)) * 20,
    "track": ["Monaco", "Monza", "Silverstone", "Spa", "Bahrain"] * 20,
    "weather": ["Sunny", "Rain", "Cloudy", "Sunny", "Rain"] * 20,
    "podium": [1, 1, 1, 0, 0] * 20
})

# Encode categories
le_driver = LabelEncoder()
le_team = LabelEncoder()
le_track = LabelEncoder()
le_weather = LabelEncoder()

data["driver"] = le_driver.fit_transform(data["driver"])
data["team"] = le_team.fit_transform(data["team"])
data["track"] = le_track.fit_transform(data["track"])
data["weather"] = le_weather.fit_transform(data["weather"])

X = data[["driver", "team", "qual_pos", "track", "weather"]]
y = data["podium"]

model = RandomForestClassifier()
model.fit(X, y)

# Save everything into a dict
model_bundle = {
    "model": model,
    "le_driver": le_driver,
    "le_team": le_team,
    "le_track": le_track,
    "le_weather": le_weather
}

with open("f1_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ Model trained and saved!")
