from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import os
from matplotlib import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "key"
app.config['UPLOAD_FOLDER'] = "static"

model = load_model("model.pkl")

@app.route("/")
def index():
    return render_template("index.html", result="waiting for input", path="mnist.png")


@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.files["file"]:
        file = request.files["file"]
    else:
        flash("please upload file")
        return redirect(url_for("index"))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    img = np.array([image.imread(file_path)])
    img = np.reshape(img, (28,28,4))
    pred = np.argmax(model.predict(img))
    print(file_path)
    return render_template("index.html", result=pred, path=file.filename)
        
if __name__ == "__main__":
    app.run(debug=True)
