import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r'C:\Users\ADMIN\Desktop\sales prediction using python\advertising.csv')

print(data.head())
print(data.isnull().sum())
print(data.describe())

sns.pairplot(data)
plt.show()

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features', fontsize=16)
plt.show()

X = data[['TV', 'Radio', 'Newspaper']]

y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
y_pred = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('ACTUAL SALES')
plt.ylabel('PREDICTED SALES')
plt.title('Actual vs Predicted Sales')
plt.show()





