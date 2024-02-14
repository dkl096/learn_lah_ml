from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("clustering_model.pkl", "rb"))

@flask_app.after_request
def add_cache_control(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

@flask_app.route('/predict-user-cluster', methods=['GET','POST'])
def predict_user_cluster():
    json_data = request.json

    try:
        acc = [float(x) for x in json_data['accuracy']]
        lis = [float(x) for x in json_data['login_streak']]
        qtt = [float(x) for x in json_data['quiz_time_taken']]
        qwl = [int(x) for x in json_data['quiz_word_learnt']]

        #Code if using URL parameter
        
        #acc = request.args.get('acc') or request.form.get('acc')
        #lis = request.args.get('lis') or request.form.get('lis')
        #qtt = request.args.get('qtt') or request.form.get('qtt')
        #qwl = request.args.get('qwl') or request.form.get('qwl')
    
        #acc = float(acc)
        #lis = float(lis)
        #qtt = float(qtt)
        #qwl = int(qwl)

    except ValueError:
        acc = 0.0
        lis = 0.0
        qtt = 0.0
        qwl = 0

    features = [acc, lis, qtt, qwl]
    features_array = np.array(features).reshape(1, -1)

    # scaler = StandardScaler()
    # scaled_features= scaler.fit_transform(features_array)

    prediction = model.predict(features_array)

    response = {'user_cluster': prediction[0]}

    return jsonify(response)
    

if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0', port=8080)