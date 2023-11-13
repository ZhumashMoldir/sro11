from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Создаем случайные данные для примера
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.randn(100)

# Разделяем данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем экземпляры различных моделей
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()
boosting_model = GradientBoostingRegressor()

# Создаем квазилинейную композицию с использованием голосования
quasilinear_model = VotingRegressor(estimators=[('linear', linear_model), ('tree', tree_model), ('boosting', boosting_model)])

# Обучаем модели
linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
boosting_model.fit(X_train, y_train)
quasilinear_model.fit(X_train, y_train)

# Делаем предсказания
linear_predictions = linear_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)
boosting_predictions = boosting_model.predict(X_test)
quasilinear_predictions = quasilinear_model.predict(X_test)

# Оцениваем качество предсказаний
print("Linear Model MSE:", mean_squared_error(y_test, linear_predictions))
print("Tree Model MSE:", mean_squared_error(y_test, tree_predictions))
print("Boosting Model MSE:", mean_squared_error(y_test, boosting_predictions))
print("Quasilinear Model MSE:", mean_squared_error(y_test, quasilinear_predictions))
