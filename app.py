from flask import Flask, render_template, request
import pickle

filename = r'models/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
count_vect = pickle.load(open('models/count_tfid', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    pred = loaded_model.predict(count_vect.transform([data1]))
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)















