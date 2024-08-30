
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

# Load pre-trained models (assuming they are already trained and persisted)
algorithms = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC()
}

# Function to handle model training (optional, if models need retraining)
def train_models(X_train, y_train):
    for name, algorithm in algorithms.items():
        algorithm.fit(X_train, y_train)
    print("Models retrained successfully.")

# Function to handle prediction requests
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Missing data in request'}), 400

        X_test = data.get('X_test')
        if X_test is None:
            return jsonify({'error': 'Missing X_test data in request'}), 400

        # Choose the model based on request parameter or default (optional)
        model_name = request.args.get('model', 'LogisticRegression')  # Default to LogisticRegression
        if model_name not in algorithms:
            return jsonify({'error': f'Invalid model name: {model_name}'}), 400

        model = algorithms[model_name]

        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON
        }

        return jsonify({'model': model_name, 'prediction': y_pred.tolist(), 'scores': scores})

    else:
        return jsonify({'error': 'Only POST requests allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production
