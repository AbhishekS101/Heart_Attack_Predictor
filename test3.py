# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter.messagebox import showinfo

root = Tk()
root.geometry('450x650')


def check(check_list):
    # Importing the dataset
    dataset = pd.read_csv('heart.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 13].values

    # # Missing Data
    # """from sklearn.preprocessing import Imputer
    # imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
    # imputer= imputer.fit(X[:, 1:3])
    # X[:, 1:3]  = imputer.transform(X[:, 1:3])"""
    #
    # # Encoding Categorical Data
    # """from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # labelencoder_X = LabelEncoder()
    # X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # onehotencoder = OneHotEncoder(categorical_features = [0])
    # X = onehotencoder.fit_transform(X).toarray()
    # labelencoder_Y = LabelEncoder()
    # Y = labelencoder_Y.fit_transform(Y)"""

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

    #adding my check_list
    check_array = np.array(check_list)
    check_array = check_array.reshape(1 , 13)
    X_train = np.append(X_train, check_array)
    X_train = X_train.reshape(204, 13)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # """sc_Y = StandardScaler()
    # Y_train = sc_Y.fit_transform(Y_train)
    # Y_test = sc_Y.fit_transform(Y_test)"""

    #extracting my check_list
    new_check_array = X_train[203,:]
    X_train = np.delete(X_train, 203, 0)
    new_check_array = new_check_array.reshape(1,13)

    #Fitting Classifier to the Training Set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
    classifier.fit(X_train, Y_train)

    #predicting the Test Set Results
    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)

    healthy = 'Dear {0} \n You are healthy person and keep it up'.format(x)
    unhealthy = 'Dear {0} \n You are unhealthy person and you may get a heart attack soo so kindly refer to a Doctor'.format(x)

    #getting results for user input
    User_pred = classifier.predict(new_check_array)
    if User_pred == 0:
        return 0
    else:
        return 1

#creating list
def entry():
    global x
    x = link.get()
    check_list = []
    check_list.append(int(link2.get()))
    check_list.append(int(link3.get()))
    check_list.append(int(link4.get()))
    check_list.append(int(link5.get()))
    check_list.append(int(link6.get()))
    check_list.append(int(link7.get()))
    check_list.append(int(link8.get()))
    check_list.append(int(link9.get()))
    check_list.append(int(link10.get()))
    check_list.append(float(link11.get()))
    check_list.append(int(link12.get()))
    check_list.append(int(link13.get()))
    check_list.append(int(link14.get()))
    healthy = 'Dear {0} \n You are healthy person and keep it up'.format(x)
    unhealthy = 'Dear {0} \n You are unhealthy person and you may get a heart attack soon so kindly refer to a Doctor'.format(x)
    if check(check_list) == 0:
        m = "Test Results for "
        m = m+x
        showinfo(m, healthy)
    else:
        m = "Test Results for "
        m = m+x
        showinfo(m, unhealthy)

    exit()

    # m = "Test Results for "
    # m = m+x
    # exit()

f = Frame(root)
f.grid()
Label(f, text='========HEART ATTACK DETECTOR========', font=30, padx=6).pack()
f1 = Frame(root)
f1.grid()
Label(f1, text='Enter Name here', font=5).grid(row=1)
Label(f1, text="Enter Age here", font=5).grid(row=2)
Label(f1, text="Enter Sex here", font=5).grid(row=3)
Label(f1, text="Enter CP here", font=5).grid(row=4)
Label(f1, text="Enter TrestBps here", font=5).grid(row=5)
Label(f1, text="Enter Cholestrol here", font=5).grid(row=6)
Label(f1, text="Enter Fbs here", font=5).grid(row=7)
Label(f1, text="Enter ECG here", font=5).grid(row=8)
Label(f1, text="Enter Thalach here", font=5).grid(row=9)
Label(f1, text="Enter Exang here", font=5).grid(row=10)
Label(f1, text="Enter OldPeak here", font=5).grid(row=11)
Label(f1, text="Enter Slope here", font=5).grid(row=12)
Label(f1, text="Enter CA here", font=5).grid(row=13)
Label(f1, text="Enter Thal here", font=5).grid(row=14)

link = StringVar()
link2 = StringVar()
link3 = StringVar()
link4 = StringVar()
link5 = StringVar()
link6 = StringVar()
link7 = StringVar()
link8 = StringVar()
link9 = StringVar()
link10 = StringVar()
link11 = StringVar()
link12 = StringVar()
link13 = StringVar()
link14 = StringVar()

e1 = Entry(f1, font=5, textvariable=link).grid(row=1, column=1, pady=5, padx=10)
e2 = Entry(f1, font=5, textvariable=link2).grid(row=2, column=1, pady=5, padx=10)
e3 = Entry(f1, font=5, textvariable=link3).grid(row=3, column=1, pady=5, padx=10)
e4 = Entry(f1, font=5, textvariable=link4).grid(row=4, column=1, pady=5, padx=10)
e5 = Entry(f1, font=5, textvariable=link5).grid(row=5, column=1, pady=5, padx=10)
e6 = Entry(f1, font=5, textvariable=link6).grid(row=6, column=1, pady=5, padx=10)
e7 = Entry(f1, font=5, textvariable=link7).grid(row=7, column=1, pady=5, padx=10)
e8 = Entry(f1, font=5, textvariable=link8).grid(row=8, column=1, pady=5, padx=10)
e9 = Entry(f1, font=5, textvariable=link9).grid(row=9, column=1, pady=5, padx=10)
e10 = Entry(f1, font=5, textvariable=link10).grid(row=10, column=1, pady=5, padx=10)
e11 = Entry(f1, font=5, textvariable=link11).grid(row=11, column=1, pady=5, padx=10)
e12 = Entry(f1, font=5, textvariable=link12).grid(row=12, column=1, pady=5, padx=10)
e13 = Entry(f1, font=5, textvariable=link13).grid(row=13, column=1, pady=5, padx=10)
e14 = Entry(f1, font=5, textvariable=link14).grid(row=14, column=1, pady=5, padx=10)

Button(f1, text='Initiate', padx=50, relief=RAISED, font=10, borderwidth=5, command=entry).grid(column=1, pady=5)


root.mainloop()