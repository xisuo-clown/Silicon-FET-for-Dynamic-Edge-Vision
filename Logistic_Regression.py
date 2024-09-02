def logistic_regression():
    import time
    import os
    import numpy as np

    # load data
    train_path = ("C:/Users/ASUS/OneDrive - Nanyang Technological University/"
                  "datasets/DVS128Gesture/event_array/polar_train_set_eve/")
    test_path = ("C:/Users/ASUS/OneDrive - Nanyang Technological University/"
                 "datasets/DVS128Gesture/event_array/polar_test_set_eve/")

    # processing in
    features_train = np.load(os.path.join(train_path, "dataset_features_remove_flatten.npy"), allow_pickle=True)
    features_test = np.load(os.path.join(test_path, "dataset_features_remove_flatten.npy"), allow_pickle=True)
    targets_train = np.load(os.path.join(train_path, "dataset_labels_remove.npy"), allow_pickle=True)
    targets_test = np.load(os.path.join(test_path, "dataset_labels_remove.npy"), allow_pickle=True)
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    features_train = std.fit_transform(features_train)
    features_test = std.fit_transform(features_test)
    # # normalization
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(features_train)
    # features_train = scaler.transform(features_train)
    # features_test = scaler.transform(features_test)

    from sklearn.utils import shuffle
    features_train, targets_train = shuffle(features_train, targets_train, random_state=41)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    # Logistic version---------------------------------
    start = time.time()
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                             intercept_scaling=1, l1_ratio=None, max_iter=100,
                             multi_class='auto', n_jobs=None, penalty='l2',
                             random_state=None,
                             # solver='saga',
                             solver='saga',
                             tol=0.0001,
                             verbose=2,
                             warm_start=False
                             )
    # clf =
    clf.fit(features_train, targets_train)
    print("classes: " + '{}'.format(clf.classes_))
    # print(clf.predict(features_test))
    # print(targets_test)
    clf.score(features_test, targets_test)  # training score
    targets_pred = clf.predict(features_test)
    accuracy = clf.score(features_test, targets_test)
    print("predict labels are" + "{}".format(targets_pred))
    print("real labels are" + "{}".format(targets_test))
    print("Accuracy of model" + "{}".format(accuracy))

    coefficients = clf.coef_
    intercept = clf.intercept_



    # Calculate the total number of parameters
    num_parameters = np.size(coefficients) + np.size(intercept)

    print("Logistic Regression Model Summary")
    print("=================================")

    print(f"Total number of parameters: {num_parameters}")
    print("Coefficients:")
    for i, class_coefficients in enumerate(coefficients):
        print(f"  Class {i}: {class_coefficients}")
    print(f"Intercept: {intercept}")

    end = time.time()
    print(end - start)
    # # -----------------SVM-------------------------------
    # from sklearn.linear_model import SGDRegressor
    # # 初始化随机梯度下降优化模型
    # start = time.time()
    # from sklearn.svm import SVC
    #
    # # import metrics to compute accuracy
    # from sklearn.metrics import accuracy_score
    # # instantiate classifier with default hyperparameters
    # svc = SVC(C=10.0,verbose=1)
    # # fit classifier to training set
    # svc.fit(features_train, targets_train)
    # # make predictions on test set
    # targets_pred = svc.predict(features_test)
    # accuracy = accuracy_score(targets_test, targets_pred)
    # # compute and print accuracy score
    # print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy))
    # end = time.time()
    # print(end - start)
    # # -----------------end linear------------------------

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    from resnet_10 import plot_cm

    plot_cm(y_test=targets_test, y_pred=targets_pred, accuracy=accuracy)

