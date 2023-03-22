from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import pickle

app = Flask(__name__)

# Load Models
with open('linearRegression.pkl', 'rb') as f:
    linear_regression_model = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
    
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
    
with open('gaussian_model.pkl', 'rb') as f:
    gaussian_model = pickle.load(f)

with open('multinomial_model.pkl', 'rb') as f:
    multinomial_model = pickle.load(f)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/salary_pred')
def salary_pred():
    return render_template('salary_pred.html')

@app.route('/knn')
def knn():
    return render_template('knn.html')

@app.route('/kmeans')
def kmeans():
    return render_template('kmeans.html')

@app.route('/gaussian')
def gaussian():
    return render_template('gaussian.html')
@app.route('/multinomial')
def multinomial():
    return render_template('multinomial.html')
@app.route('/bernouli')
def bernouli():
    return render_template('bernouli.html')

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    select_ = float(request.form.get('pos_lvl'))
    exp_ = float(request.form.get('experience'))
    if np.isnan(select_) or np.isinf(select_) or np.isnan(exp_) or np.isinf(exp_):
        return Response("Invalid input values", status=400)
    int_features = [select_, exp_]
    final_features = np.array([int_features])
    prediction = linear_regression_model.predict(final_features)
    res = prediction.item()
    return render_template('salary_pred.html', prediction_text='Expected Salary Rate should be ${:.2f}'.format(res))

@app.route('/knn_predict', methods=['POST'])
def knn_predict():
    salary = float(request.form.get('salary'))
    exp_ = float(request.form.get('experience'))
    if np.isnan(salary) or np.isinf(salary) or np.isnan(exp_) or np.isinf(exp_):
        return Response("Invalid input values", status=400)
    int_features = [exp_, salary]
    final_features = np.array([int_features])
    prediction = knn_model.predict(final_features)
    job_level_number = prediction.item()
    job_level_dict = {
    1: "Junior",
    2: "Senior",
    3: "Project Manager",
    4: "CTO"
}
    job_level_name = job_level_dict.get(job_level_number)
    return render_template('knn.html', prediction_text='Expected Job Level is: {}'.format(job_level_name))

@app.route('/kmeans_predict', methods=['POST'])
def kmeans_predict():
    salary = float(request.form.get('salary'))
    exp_ = float(request.form.get('experience'))
    if np.isnan(salary) or np.isinf(salary) or np.isnan(exp_) or np.isinf(exp_):
        return Response("Invalid input values", status=400)
    int_features = [exp_, salary]
    final_features = np.array([int_features])
    cluster = kmeans_model.predict(final_features)[0]
    job_level_dict = {0: 'Junior', 1: 'Senior', 2: 'Project Manager', 3: 'CTO'}
    job_level_name = job_level_dict.get(cluster)
    return render_template('kmeans.html', prediction_text='Expected Job Level is: {}'.format(job_level_name))

@app.route('/gaussian_predict', methods=['POST'])
def gaussian_predict():
    salary= float(request.form.get('salary'))
    exp_ = float(request.form.get('experience'))
    if np.isnan(salary) or np.isinf(salary) or np.isnan(exp_) or np.isinf(exp_):
        return Response("Invalid input values", status=400)
    int_features = [exp_, salary]
    final_features = np.array([int_features])
    prediction = gaussian_model.predict(final_features)
    job_level_number = prediction.item()
    job_level_dict = {
    1: "Junior",
    2: "Senior",
    3: "Project Manager",
    4: "CTO"
    }
    job_level_name = job_level_dict.get(job_level_number)
    return render_template('gaussian.html', prediction_text='Expected Job Level is: {}'.format(job_level_name))

@app.route('/multinomial_predict', methods=['POST'])
def multinomial_predict():
    salary= float(request.form.get('salary'))
    exp_ = float(request.form.get('experience'))
    if np.isnan(salary) or np.isinf(salary) or np.isnan(exp_) or np.isinf(exp_):
        return Response("Invalid input values", status=400)
    int_features = [exp_, salary]
    final_features = np.array([int_features])
    prediction = gaussian_model.predict(final_features)
    job_level_number = prediction.item()
    job_level_dict = {
    1: "Junior",
    2: "Senior",
    3: "Project Manager",
    4: "CTO"
    }
    job_level_name = job_level_dict.get(job_level_number)
    return render_template('multinomial.html', prediction_text='Expected Job Level is: {}'.format(job_level_name))

@app.route('/bernouli_predict', methods=['POST'])
def bernouli_predict():
    salary= float(request.form.get('salary'))
    exp_ = float(request.form.get('experience'))
    if np.isnan(salary) or np.isinf(salary) or np.isnan(exp_) or np.isinf(exp_):
        return Response("Invalid input values", status=400)
    int_features = [exp_, salary]
    final_features = np.array([int_features])
    prediction = gaussian_model.predict(final_features)
    job_level_number = prediction.item()
    job_level_dict = {
    1: "Junior",
    2: "Senior",
    3: "Project Manager",
    4: "CTO"
    }
    job_level_name = job_level_dict.get(job_level_number)
    return render_template('bernouli.html', prediction_text='Expected Job Level is: {}'.format(job_level_name))

if __name__ == '__main__':
    app.run(debug=True)
