from flask import Flask, render_template, request
import pickle as pkl


# Flask App Here
app = Flask(__name__)

# Home Page
@app.route('/')
def home():
    return render_template("index.html")

def getData():
    gender = int(request.form.get("gender"))
    age = int(request.form.get("age"))
    smoke = int(request.form.get("smoke"))
    yellow_finger = int(request.form.get("yellow_finger"))
    anxiety = int(request.form.get("anxiety"))
    peer_pressure = int(request.form.get("peer_pressure"))
    chronic_disease = int(request.form.get("chronic_disease"))
    fatigue = int(request.form.get("fatigue"))
    allergy = int(request.form.get("allergy"))
    wheezing = int(request.form.get("wheezing"))
    alcohol = int(request.form.get("alcohol"))
    cough = int(request.form.get("cough"))
    shortness_in_breath = int(request.form.get("shortness_in_breath"))
    swallowing = int(request.form.get("swallowing"))
    chest_pain = int(request.form.get("chest_pain"))

    X_pred = [[gender, age, smoke, yellow_finger, anxiety,
                peer_pressure, chronic_disease, fatigue,
               allergy, wheezing, alcohol, cough, shortness_in_breath,
               swallowing, chest_pain]]

    print(X_pred)
    return X_pred

def modelPredict(X_pred):
    with open("/Users/Awan Hashir/Downloads/AI projects/Internship/lung-cancer 2/models/LogisticRegressionModel.pkl", "rb") as f:
        model = pkl.load(f)
    y_pred = model.predict(X_pred)
    print(y_pred)
    return y_pred

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        data = getData()
        output = int(modelPredict(data)[0])
        if output == 1:
            return render_template('predict.html', prediction="You may have Cancer .")
        else:
            return render_template('predict.html', prediction="You may not have Cancer .")
    except ValueError:
        return render_template("none.html") 
   

if __name__ == "__main__":
    app.run(debug=True, port="80")