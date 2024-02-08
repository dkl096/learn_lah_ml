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

@flask_app.route('/predict_user_cluster', methods=['GET','POST'])
def predict_user_cluster():
    # json_data = request.json_data
    # acc = [float(x) for x in json_data['accuracy']]
    # lis = [float(x) for x in json_data['login_streak']]
    # qtt = [float(x) for x in json_data['quiz_time_taken']]
    # qwl = [int(x) for x in json_data['quiz_word_learnt']]

    acc = request.args.get('acc') or request.form.get('acc')
    lis = request.args.get('lis') or request.form.get('lis')
    qtt = request.args.get('qtt') or request.form.get('qtt')
    qwl = request.args.get('qwl') or request.form.get('qwl')

    try:
        acc = float(acc)
        lis = float(lis)
        qtt = float(qtt)
        qwl = int(qwl)
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
    prediction_lower_case = prediction[0].lower()

    # response = {'user_cluster': prediction[0]}

    # return jsonify(response)

    prediction_lower_case = prediction[0].lower()
    if prediction_lower_case == 'poor':
        return "Your progress is lacking. Please do more!"
    elif prediction_lower_case == 'beginner':
        return "First step is everything. Keep going!"
    elif prediction_lower_case == 'average':
        return "You are faring better than 75% of user. Keep going!"
    else:
        return "Amazing! Keep it up!"
    

if __name__ == '__main__':
    flask_app.run(debug=True, port=8080)
    #serve(app, host="0.0.0.0", port=8080)