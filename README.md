# Fraudulent-Baking-Transaction-Recognition
Classifying the Fradulent and Non-Fradulent transactions using  Classification ALgorithms


Fraudulent Banking Operations Recognition - Machine Learning Model
This project investigates fraudulent banking operations using machine learning algorithms. It references the published journal "Developing AI-based Fraud Detection Systems for Banking and Finance" 

# Dataset:

Credit Card transactions dataset (details anonymized due to security concerns)
Data is already standardized.

# Steps Followed:

# Feature Engineering:

## Data Cleaning: No missing values present due to pre-processing.
Feature Selection: All features except "time" are included.
Dimensionality Reduction: PCA (Principal Component Analysis) is not applied as sufficient features remain after initial processing.
Feature Scaling: StandardScaler is applied to the "Amount" column due to its unit (dollars).
Imbalanced Data Handling:

The dataset is imbalanced (unequal distribution of fraudulent and legitimate transactions).
Undersampling is used to balance the data (reducing the majority class).
# Data Splitting:
Independent features (predictors) and dependent feature (target variable) are separated.
# Training and Testing Split:
Data is split into training (80%) and testing (20%) sets.
Machine Learning Model Selection:
Logistic Regression

Decision Tree Classifier

Random Forest

Support Vector Machine
# Choosing the Best Model
Random Forest achieved the best performance among the evaluated algorithms with accuracy of 96 % .


# 1. Imports:

Flask: Used to create the web application.
render_template: Renders HTML templates for user interaction.
request: Handles incoming HTTP requests from the client.
jsonify: Converts Python data structures to JSON format for responses.
Machine learning libraries (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix): Used for training and evaluating the models (assuming these are already trained and persisted).
# 2. Flask Application Initialization:

app = Flask(__name__): Creates a Flask application instance named app.
algorithms: A dictionary that stores pre-trained machine learning models.
# 3. Training Function (Optional):

train_models(X_train, y_train): This function is commented out but could be used to retrain the models if needed. It iterates through the algorithms dictionary and calls the fit() method on each model to train it on the provided training data (X_train and y_train).
# 4. Routes:

# @app.route('/'): This decorator defines a route for the root URL (/).

return render_template('index.html'): Renders the index.html template for the main page, which likely provides a user interface for entering data and selecting the model.
# @app.route('/predict', methods=['POST']): This decorator defines a route for the /predict endpoint and specifies that it only accepts POST requests.

data = request.get_json(): Retrieves the data sent in the request body as JSON.
Error handling for missing data: Checks if the request is missing data and returns an error response (400 Bad Request) if so.
X_test = data.get('X_test'): Extracts the X_test data from the JSON request.
Model selection:
model_name = request.args.get('model', 'LogisticRegression'): Retrieves the model name from the request parameters, defaulting to "LogisticRegression" if not specified.
Error handling for invalid model name.
Prediction:
model = algorithms[model_name]: Selects the chosen machine learning model from the algorithms dictionary.
y_pred = model.predict(X_test): Makes predictions using the chosen model on the X_test data.
Evaluation:
scores dictionary: Calculates various evaluation metrics like accuracy, F1-score, precision, recall, and confusion matrix using the missing y_test data (replace with your actual test data).
Successful response: Returns JSON data containing the model name, prediction results, and evaluation scores.
# 5. Running the Application:

if __name__ == '__main__':: Ensures the code within this block only runs when the script is executed directly (not imported as a module).
app.run(debug=True): Starts the Flask application in debug mode for development. Set debug=False for production deployments.

