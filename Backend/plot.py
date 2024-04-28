import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.scatter(y_test, voting_model.predict(X_test), color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# Residual plot
plt.figure(figsize=(12, 6))
sns.residplot(y_test, voting_model.predict(X_test), lowess=True, color='g')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Feature importance plot
feature_importance = pd.Series(voting_model.feature_importances_, index=features)
feature_importance.nlargest(10).plot(kind='barh', figsize=(12, 6))
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top 10 Important Features')
plt.show()
