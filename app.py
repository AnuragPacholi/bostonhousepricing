# import sklearn
import pickle
from typing import final
from flask import Flask, json,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__) # Starting point of the application.

regmodel = pickle.load(open('regmodel.pkl','rb')) # Opening and loading our model.
scalar = pickle.load(open('scaling.pkl','rb')) # Opening and loading our stardizing pickel file.

@app.route('/') # Starting point of our app.
def home(): # This will load our homepage. (i.e. loading our html file which we made for homepage.)
    return render_template('home.html') # In our homepage we'll display our home's html file.
    # This 'render_template' will redirect us to the file that we have passed in it.
    # It looks for that file in the 'templates' folder(which we have made) and execute it.

# Making an API through which we can get the input from the app, send it to our model, predict the result and send it back to the app.
@app.route('/predict_api',methods=['POST']) # We are using methods == Post because we want to post/send the input to our model.
def predict_api(): # Making the API.
    data=request.json['data'] # Here in this line we are basically taking the input and storing it into the variable named as 'data'.
    # We are requesting the input in json format. And then we store it into the 'data' variable.
    print(data) # Just to check our inputs.
    print(np.array(list(data.values())).reshape(1,-1)) # Just to check our reshaped inputs.
    # Since json is a key-value pair format, so data.values() gives us the values. And then we're putting values into a list.
    # Since we need to reshape the data as we saw in ipynb notebook therefore we're converting the list to np array and reshaping as we did in ipynb.
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    # Stndardizing the input from the scalar pickle that we loaded.
    output = regmodel.predict(new_data) # Prediction
    print(output[0]) # The prediction that we get is in 2D with only one value, therefore printing that one value. (to check)
    return jsonify(output[0]) # Returning the prediction.

    # As you can notice in this API we are doing every major thing, we are reshaping the input, sending it, predicting from the model and getting the output.
    # We'll also use Postman to test this API.


# WITHOUT USING AN API:
@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    # Here in the above line we are capturing all the values that we got from out html form(i.e. homepage of out app).
    # So basically we are requesting for those input values in a list. We want them in float type.
    final_input = scalar.transform(np.array(data).reshape(1,-1)) # Standardizing and reshaping.
    print(final_input)
    output = regmodel.predict(final_input)[0] # Prediction
    return render_template("home.html", prediction_text = 'The Predicted House Price is {}'.format(output))
    # In the above line we are returning a template and in it we'll have a place holder i.e. our prediction text which also contains the output value using the '.format'.
    # The template which we are returning above is the same as our homepage. That means we'll show the result in the same page. You can show a different page also, just make a template/html page for it.

# THIS ABOVE PREDICTION FUNCTION COULD WORK WITHOUT THE API THAT WE BUILT PERHAPS.



if __name__=="__main__": # Needed to run the app.
    app.run(debug=True)