from flask import Flask, render_template, request
import pickle

import numpy as np

app = Flask(__name__,template_folder='.')

rf_model=pickle.load(open('model_rf','rb'))

@app.route('/',methods=['GET'])
def main():
	return render_template("pcos detection.html")

@app.route('/', methods=['POST'])
def predict():
	if request.method == 'POST':

		final=[]
		final.append(float(request.form['skindarkening']))
		final.append(float(request.form['hairgrowth']))
		final.append(float(request.form['weightgain']))
		final.append(float(request.form['cycle']))
		#final.append(float(request.form['pimple']))

		req_model = str(request.form['model'])

		if req_model == "Random Forest Regressor":
			final = np.array(final)
			arr = final.reshape(1, -1)
			prediction = rf_model.predict(arr)
			prediction = prediction[0]

		if prediction == 0:
			return render_template("pcos detection.html",pred ='You do not have significant chances of having PCOS. However, if you are experiencing some other symptoms or feeling that you need a diagnosis, it is good to bring it up in your next doctor\'s appointment.')

		if prediction == 1:
			return render_template("pcos detection.html",pred ='You have significant chances of having PCOS. Please check with your doctor as soon as possible.')


if __name__ == "__main__":
	app.run(debug=True)