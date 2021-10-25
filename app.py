from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stat
from scipy.special import inv_boxcox1p

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
enc = pickle.load(open('encoder.pickle', 'rb'))
lamda_dict_boxcox_transformation = {
    'age': 0.6018414717768876,
    'fpl_points': 0.3416036419202748,
    'fpl_sel': -0.677051279106644,
    'market_value': -0.01790074529048561,
    'page_views': 0.0666495005672392
    }

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    global lamda_dict_boxcox_transformation
    if request.method == 'POST':
        age = request.form['age']
        position = request.form['position']
        position_cat = request.form['position_cat']
        page_views = request.form['page_views']
        fpl_value = request.form['fpl_value']
        fpl_sel = request.form['flp_sel']
        fpl_points = request.form['fpl_points']
        region = request.form['region']
        new_foreign = request.form['new_foreign']
        age_cat = request.form['age_cat']
        club_id = request.form['club_id']
        big_club = request.form['big_club']
        new_sining = request.form['new_signing']
        print(age)
        print(position)
        print(position_cat)
        print(page_views)
        print(fpl_value)
        print(fpl_sel)
        print(fpl_points)
        print(region)
        print(new_foreign)
        print(age_cat)
        print(club_id)
        print(big_club)
        print(new_sining)
        
        continuous_feature = {
            'age':float(age), 
            'page_views':float(page_views), 
            'fpl_sel':float(fpl_sel), 
            'fpl_points':float(fpl_points)
            }
        
        for lable in continuous_feature.keys():
            continuous_feature[lable]= stat.boxcox([continuous_feature[lable] + 1, 1111], lamda_dict_boxcox_transformation[lable])[0]
        
        data = [
            continuous_feature['age'],
            float(position_cat),
            continuous_feature['page_views'],
            float(fpl_value),
            continuous_feature['fpl_sel'],
            continuous_feature['fpl_points'],
            float(region),
            float(new_foreign),
            float(age_cat),
            float(club_id),
            float(big_club),
            float(new_sining),
        ]
        encoded__pos_vector = list(enc.transform([[str(position)]]).toarray()[0])
        data_vector = data + encoded__pos_vector[1:]
        prediction = model.predict([data_vector])
        output = inv_boxcox1p(prediction, lamda_dict_boxcox_transformation['market_value'])
        return render_template('index.html',prediction_text="Player Price is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

