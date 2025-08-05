import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Load data
data = pd.read_csv("student_scores.csv")

# Step 2: Normalize (optional if your max is already 24 and 100)
data['Hours'] = data['Hours'] / 24.0
data['Scores'] = data['Scores'] / 100.0

# Step 3: Split data
X = data[['Hours']]   # input feature
y = data['Scores']    # target output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
hours = float(input("Enter study hours (0 to 12): "))
normalized_hours = hours / 12.0
predicted_score = model.predict([[normalized_hours]])[0]
actual_score = predicted_score * 100

print(f"Predicted Marks for {hours} hours study = {actual_score:.2f} out of 100")

# Step 6: Optional - Plotting
plt.scatter(X * 12, y * 100, color='blue', label='Actual')
plt.plot(X * 12, model.predict(X) * 100, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks (%)')
plt.title('Student Marks Prediction')
plt.legend()
plt.grid(True)
plt.show()

