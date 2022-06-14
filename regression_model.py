from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, \
    AdaBoostRegressor, GradientBoostingRegressor


def fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users, model='lr'):
    '''
    :param X_train_users: train_x
    :param Y_train_users: train_Y
    :param X_test_users: test_x
    :param Y_test_users: test_y
    :param model: "lr" for linear regression, "rf" for random forest, "nn" for neural network
    :return: predicted values, r2_score
    '''
    if model == "lr":
        return lr_fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users)
    elif model == "rf":
        return rf_fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users)
    elif model == "ada":
        return ada_fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users)


def reg_fit(X_train_users, Y_train_users, model='lr'):
    '''
    :param X_train_users: train_x
    :param Y_train_users: train_Y
    :param model: "lr" for linear regression, "rf" for random forest, "nn" for neural network
    :return: model
    '''
    if model == "lr":
        return lr_fit(X_train_users, Y_train_users)
    elif model == "rf":
        return rf_fit(X_train_users, Y_train_users)
    elif model == "ada":
        return ada_fit(X_train_users, Y_train_users)


def lr_fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users):
    model = LinearRegression()
    model.fit(X_train_users, Y_train_users)
    y_pred_users = model.predict(X_test_users)
    lr_score = model.score(X_test_users, Y_test_users)
    return model, y_pred_users, lr_score


def rf_fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users):
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(X_train_users, Y_train_users)
    y_pred_users = model.predict(X_test_users)
    rf_score = model.score(X_test_users, Y_test_users)
    return model, y_pred_users, rf_score


def ada_fit_predict(X_train_users, Y_train_users, X_test_users, Y_test_users):
    model = AdaBoostRegressor(n_estimators=1000)
    model.fit(X_train_users, Y_train_users)
    y_pred_users = model.predict(X_test_users)
    rf_score = model.score(X_test_users, Y_test_users)
    return model, y_pred_users, rf_score


def lr_fit(X_train_users, Y_train_users):
    model = LinearRegression()
    model.fit(X_train_users, Y_train_users)
    return model


def rf_fit(X_train_users, Y_train_users):
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train_users, Y_train_users)
    return model


def ada_fit(X_train_users, Y_train_users):
    model = AdaBoostRegressor(n_estimators=1000)
    model.fit(X_train_users, Y_train_users)
    return model


def gb_fit(X_train_users, Y_train_users):
    model = GradientBoostingRegressor(n_estimators=1000)
    model.fit(X_train_users, Y_train_users)
    return model
