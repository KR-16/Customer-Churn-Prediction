binary_class_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"
]

multi_class_columns = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]

numerical_columns = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"
]

target_column = "Churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_NAMES = ["Logistic_Regression", "Random_Forest", "KNN", "SVM", "XG_Boost"]
MAXIMUM_ITERATIONS = 1000
N_ESTIMATORS = 100
CLASS_WEIGHT = "balanced"
USE_LABEL_ENCODER = False
EVAL_METRIC = "logloss"
SCALE_POS_WEIGHT = 1
PROBABILITY = True