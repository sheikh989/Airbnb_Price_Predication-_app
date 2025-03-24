import joblib
import pickle

model_path = "APP/airbnb_model.pkl"


with open(model_path, "rb") as file1:
    airbnb_model = pickle.load(file1)
joblib.dump(airbnb_model, "airbnb_model.joblib")

with open("APP/scaler_airbnb_right.pkl", "rb") as file2:
    scaler_airbnb_right = pickle.load(file2)
joblib.dump(scaler_airbnb_right, "scaler_airbnb_right.joblib")


with open("APP/city_labeling.pkl", "rb") as file3:
    city = pickle.load(file3)
joblib.dump(city, "city.joblib")

with open("APP/zip_labeling.pkl", "rb") as file4:
    zip = pickle.load(file4)
joblib.dump(zip, "zip.joblib")

with open("APP/pt_labeling.pkl", "rb") as file5:
    pt = pickle.load(file5)
joblib.dump(pt, "pt.joblib")

with open("APP/rt_labeling.pkl", "rb") as file6:
    rt = pickle.load(file6)
joblib.dump(rt, "rt.joblib")

with open("APP/lg_labeling.pkl", "rb") as file7:
    lg = pickle.load(file7)
joblib.dump(lg, "lg.joblib")

with open("APP/bt_labeling.pkl", "rb") as file8:
    bt = pickle.load(file8)
joblib.dump(bt, "bt.joblib")
