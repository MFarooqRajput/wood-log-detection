import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import neighbors

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from src.binary_classification_visual import heat_map
from src.binary_classification_visual import plot_accuracy

def df_train_test_split(df):
    columns = ['image', 'algo']
    training_percent = 0.9
    y_train, y_test, X_train, X_test  = train_test_split(df[["Damage"]], df.drop(columns=columns), train_size = training_percent)

    return  y_train, y_test, X_train, X_test

def check_correctness(df, original_label, predicted_label):
    correct = len(df[(df[original_label] == df[predicted_label])])
    incorrect = len(df) - correct

    return (correct,incorrect,(correct*100/(correct+incorrect)))

def logistic_regression(y_train, y_test, X_train, X_test):
    model = LogisticRegression(solver='liblinear', C=25.0, random_state=0).fit(X_train, y_train)

    print('Classes: ',model.classes_)
    print("\n")
    print('Intercept: ', model.intercept_)
    print("\n")
    print('Coefficeients: ', model.coef_)
    print("\n")
    
    prob = model.predict_proba(X_test)

    # Find maximum value from probability prediction array
    max_values = pd.DataFrame(prob, columns = range(0,2))

    # Dominant
    dominant = model.classes_[max_values.idxmax(axis=1, skipna=True)]
    comparison = pd.DataFrame({'Original': y_test.Damage, 'Dominant':dominant})
    print(comparison)
    print("\n")
    
    print(classification_report(y_test, model.predict(X_test), digits = 3))
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy:' + str(model_accuracy))
    print("\n")
    
    c,e,p = check_correctness(comparison, "Original", "Dominant")

    print("correct:" + str(c))
    print("incorrect:" + str(e))
    print("percentage:" + str(p))
    print("\n")
    
    confusion_matrix = comparison.groupby(['Dominant','Original']).size().unstack('Original').fillna(0)
    print(confusion_matrix)
    print("\n")
    
    pivot = confusion_matrix.to_numpy().max()/2
    heat_map(confusion_matrix, pivot, 0)
    
    return model_accuracy

def gaussiannb(y_train, y_test, X_train, X_test):
    gnb = GaussianNB()
    model = gnb.fit(X_train,y_train)
    
    prob = model.predict_proba(X_test)

    # Find maximum value from probability prediction array
    max_values = pd.DataFrame(prob, columns = range(0,2))

    # Dominant
    dominant = model.classes_[max_values.idxmax(axis=1, skipna=True)]
    comparison = pd.DataFrame({'Original': y_test.Damage, 'Dominant':dominant})
    print(comparison)
    print("\n")
    
    print(classification_report(y_test, model.predict(X_test), digits = 3))
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy:' + str(model_accuracy))
    print("\n")

    c,e,p = check_correctness(comparison, "Original", "Dominant")

    print("correct:" + str(c))
    print("incorrect:" + str(e))
    print("percentage:" + str(p))
    print("\n")
    
    confusion_matrix = comparison.groupby(['Dominant','Original']).size().unstack('Original').fillna(0)
    print(confusion_matrix)
    print("\n")
    
    pivot = confusion_matrix.to_numpy().max()/2
    heat_map(confusion_matrix, pivot, 0)
    
    return model_accuracy

def multinomialnb(y_train, y_test, X_train, X_test):
    mnb = MultinomialNB()
    model = mnb.fit(X_train,y_train)
    
    prob = model.predict_proba(X_test)

    # Find maximum value from probability prediction array
    max_values = pd.DataFrame(prob, columns = range(0,2))

    # Dominant
    dominant = model.classes_[max_values.idxmax(axis=1, skipna=True)]
    comparison = pd.DataFrame({'Original': y_test.Damage, 'Dominant':dominant})
    print(comparison)
    print("\n")
    
    print(classification_report(y_test, model.predict(X_test), digits = 3))
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy:' + str(model_accuracy))
    print("\n")

    c,e,p = check_correctness(comparison, "Original", "Dominant")

    print("correct:" + str(c))
    print("incorrect:" + str(e))
    print("percentage:" + str(p))
    print("\n")
    
    confusion_matrix = comparison.groupby(['Dominant','Original']).size().unstack('Original').fillna(0)
    print(confusion_matrix)
    print("\n")
    
    pivot = confusion_matrix.to_numpy().max()/2
    heat_map(confusion_matrix, pivot, 0)
    
    return model_accuracy

def lda(y_train, y_test, X_train, X_test):
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)

    print(model.classes_)
    print("\n")
    print(model.priors_)
    print("\n")
    print(model.means_)
    print("\n")
    print(model.coef_)
    print("\n")
    
    prob = model.predict_proba(X_test)

    # Find maximum value from probability prediction array
    max_values = pd.DataFrame(prob, columns = range(0,2))

    # Dominant
    dominant = model.classes_[max_values.idxmax(axis=1, skipna=True)]
    comparison = pd.DataFrame({'Original': y_test.Damage, 'Dominant':dominant})
    print(comparison)
    print("\n")
    
    print(classification_report(y_test, model.predict(X_test), digits = 3))
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy:' + str(model_accuracy))
    print("\n")
    
    c,e,p = check_correctness(comparison, "Original", "Dominant")

    print("correct:" + str(c))
    print("incorrect:" + str(e))
    print("percentage:" + str(p))
    print("\n")
    
    confusion_matrix = comparison.groupby(['Dominant','Original']).size().unstack('Original').fillna(0)
    print(confusion_matrix)
    print("\n")
    
    pivot = confusion_matrix.to_numpy().max()/2
    heat_map(confusion_matrix, pivot, 0)
    
    return model_accuracy

def qda(y_train, y_test, X_train, X_test):
    model = QuadraticDiscriminantAnalysis().fit(X_train, y_train)

    print(model.priors_)
    print("\n")
    print(model.means_)
    print("\n")
    
    prob = model.predict_proba(X_test)

    # Find maximum value from probability prediction array
    max_values = pd.DataFrame(prob, columns = range(0,2))

    # Dominant
    dominant = model.classes_[max_values.idxmax(axis=1, skipna=True)]
    comparison = pd.DataFrame({'Original': y_test.Damage, 'Dominant':dominant})
    print(comparison)
    print("\n")
    
    print(classification_report(y_test, model.predict(X_test), digits = 3))
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy:' + str(model_accuracy))
    print("\n")
    
    c,e,p = check_correctness(comparison, "Original", "Dominant")

    print("correct:" + str(c))
    print("incorrect:" + str(e))
    print("percentage:" + str(p))
    print("\n")
    
    confusion_matrix = comparison.groupby(['Dominant','Original']).size().unstack('Original').fillna(0)
    print(confusion_matrix)
    print("\n")
    
    pivot = confusion_matrix.to_numpy().max()/2
    heat_map(confusion_matrix, pivot, 0)
    
    return model_accuracy

def knn(y_train, y_test, X_train, X_test, n):
    model = neighbors.KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    
    prob = model.predict_proba(X_test)

    # Find maximum value from probability prediction array
    max_values = pd.DataFrame(prob, columns = range(0,2))

    # Dominant
    dominant = model.classes_[max_values.idxmax(axis=1, skipna=True)]
    comparison = pd.DataFrame({'Original': y_test.Damage, 'Dominant':dominant})
    print(comparison)
    print("\n")

    print(classification_report(y_test, model.predict(X_test), digits = 3))
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy:' + str(model_accuracy))
    print("\n")
    
    c,e,p = check_correctness(comparison, "Original", "Dominant")

    print("correct:" + str(c))
    print("incorrect:" + str(e))
    print("percentage:" + str(p))
    print("\n")
    
    confusion_matrix = comparison.groupby(['Dominant','Original']).size().unstack('Original').fillna(0)
    print(confusion_matrix)
    print("\n")

    pivot = confusion_matrix.to_numpy().max()/2
    heat_map(confusion_matrix, pivot, 0)
    
    return model_accuracy

def accuracy(model_accuracy):
    print("Logistic Regression: " + str(model_accuracy[0]))
    print("Gaussian Naïve Bayes: " + str(model_accuracy[1]))
    print("Multinomial Naïve Bayes: " + str(model_accuracy[2]))
    print("LDA: " + str(model_accuracy[3]))
    print("QDA: " + str(model_accuracy[4]))
    print("kNN with K = 1: " + str(model_accuracy[5]))
    print("kNN with K = 2: " + str(model_accuracy[6]))
    print("kNN with K = 3: " + str(model_accuracy[7]))
    print("kNN with K = 4: " + str(model_accuracy[8]))
    print("kNN with K = 5: " + str(model_accuracy[9]))
    
    col = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    accuracy = model_accuracy
    tick_label = ['Logistic Regression', 'Gaussian NB', 'Multinomial NB', 'LDA', 'QDA', 'KNN1', 'KNN2','KNN3','KNN4', 'KNN5']
    plot_accuracy(col, accuracy, tick_label)