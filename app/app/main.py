import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
                            recall_score, precision_score, roc_auc_score


def readData(filepath='data/water_potability.csv'):
    df = pd.read_csv(filepath)
    X = df.iloc[:,0:len(df.columns)-1]
    y = df.iloc[:, -1]
    return X, y


def scaleData(df):
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    
    path = 'result/scaler/MinMaxScaler.sav'
    pickle.dump(scaler, open(path, 'wb'))
    print(f'Scaler saved successfully!')
    
    df = scaler.transform(df)
    return df


def trainAndSaveModels(model, X_train, y_train):
    model_name = str(model)[ : str(model).index('(')]
    model.fit(X_train, y_train)
    path = f'result/models/{model_name}.sav'
    pickle.dump(model, open(path, 'wb'))
    print(f'{model_name} trained and saved successfully!')


def trainingPhase(X_train, y_train):
    model_list = [
        DecisionTreeClassifier(criterion='entropy', random_state=0),
        KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        LogisticRegression(random_state = 0),
        MultinomialNB(),
        RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
        SVC(kernel='linear', random_state=0)
    ]

    for model in model_list:
        trainAndSaveModels(model, X_train, y_train)


def prediction(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_acc = roc_auc_score(y_test, y_pred)
    
    return [tn, fp, fn, tp, acc, precision, recall, f1, roc_acc]


def graphResults(filepath='result/model_evaluation/ModelEvaluationScores.csv'):
    test_results = pd.read_csv('result/model_evaluation/ModelEvaluationScores.csv')
    
    classifier_abbr = []
    for classifier_name in test_results['Classifier']:
        classifier_abbr.append(('').join([
            char for char in classifier_name if char.isupper()
        ]))
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(classifier_abbr, test_results['Accuracy'], label='Accuracy')
    plt.plot(classifier_abbr, test_results['Precision'], label='Precision')
    plt.plot(classifier_abbr, test_results['Recall'], label='Recall')
    plt.plot(classifier_abbr, test_results['F1 Score'], label='F1 Score')
    plt.plot(classifier_abbr, test_results['ROC'], label='ROC Score')
    plt.xlabel('Classifier Used')
    plt.ylabel('Evaluation Score')
    plt.legend()

    bar_width = 0.20
    bar_tn = np.arange(len(classifier_abbr))
    bar_fp = [x + bar_width for x in bar_tn]
    bar_fn = [x + bar_width for x in bar_fp]
    bar_tp = [x + bar_width for x in bar_fn]
    plt.subplot(1, 2, 2)
    plt.bar(bar_tn, test_results['True Negative'], width=bar_width, color='r', label='TN')
    plt.bar(bar_fp, test_results['False Positive'], width=bar_width, color='b', label='FP')
    plt.bar(bar_fn, test_results['False Negative'], width=bar_width, color='y', label='FN')
    plt.bar(bar_tp, test_results['True Positive'], width=bar_width, color='g', label='TP')
    plt.xlabel('Classifier Used')
    plt.ylabel('Number of Samples')
    plt.xticks(
        [r + bar_width for r in range(len(classifier_abbr))],
        classifier_abbr
    )
    plt.legend()
    plt.show()


def testingPhase(X_test, y_test):
    model_list=[]
    test_results = []

    with open(f'result/scaler/MinMaxScaler.sav', 'rb') as f:
        scaler = pickle.load(f)
        X_test = scaler.transform(X_test)

    model_files = os.listdir('result/models')
    for model_file in model_files:
        with open(f'result/models/{model_file}', 'rb') as f:
            model = pickle.load(f)
            model_list.append(model)
            test_results.append([str(model)[ : str(model).index('(')]])
    
    for index, classifier in enumerate(model_list):
        scores = prediction(classifier, X_test, y_test)
        test_results[index] = test_results[index] + scores

    test_results = pd.DataFrame(test_results, columns=[
        'Classifier',
        'True Negative',
        'False Positive',
        'False Negative',
        'True Positive',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'ROC'
    ])
    test_results.to_csv('result/model_evaluation/ModelEvaluationScores.csv')


def getUserInput(data_features):
    user_input = []
    for feature in data_features.columns:
        print(f'Enter value for {feature}')
        print(f'Range is {min(data_features[feature])} to {max(data_features[feature])}')
        user_input.append(input('Your input: '))
    return user_input
    

def evaluateUserInput(data_features):
    X_test = getUserInput(data_features)
    with open(f'result/scaler/MinMaxScaler.sav', 'rb') as f:
        scaler = pickle.load(f)
        X_test = scaler.transform([X_test])

    model_accuracy_scores = pd.read_csv(
        'result/model_evaluation/ModelEvaluationScores.csv'
    )['Accuracy']

    model_list=[]
    test_results = []
    prediction_score = 0
    model_files = os.listdir('result/models')
    for model_file in model_files:
        with open(f'result/models/{model_file}', 'rb') as f:
            model = pickle.load(f)
            model_list.append(model)
            test_results.append([str(model)[ : str(model).index('(')]])

    for index, classifier in enumerate(model_list):
        y_pred = classifier.predict(X_test)
        y_pred = 'Positive' if y_pred == 1 else 'Negative'

        if y_pred == 'Positive':
            prediction_score += 1 * model_accuracy_scores[index]
        else:
            prediction_score -= 1 * model_accuracy_scores[index]

        test_results[index].append(y_pred)

    
    prediction_score = prediction_score / sum(model_accuracy_scores)
    if prediction_score < 0:
        prediction_class = 'Negative'
    elif prediction_score > 0:
        prediction_class = 'Positive'
    else:
        prediction_class = 'Indeterminate'

    
    print('\n\nResults :-')
    for test_result in test_results:
        print(test_result[0], ':', test_result[1])
    print('\n')
    print(f'The weighted average of all the models: {prediction_score}')
    print(f'The predicted label for given review is {prediction_class}')
    print('\n\n')


def main():
    X, y = readData(filepath='data/water_potability.csv')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_train = scaleData(X_train)
    # trainingPhase(X_train, y_train)
    # testingPhase(X_test, y_test)
    evaluateUserInput(X)
    # graphResults()



if __name__ == '__main__':
    main()