from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report ,confusion_matrix
import pandas as pd
import joblib



data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)  # make DataFrame
df['Target'] = data.target                                # add Target to DataFrame



X = df.drop('Target' , axis = 1 )
y = df['Target']


# Split data
x_train, x_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 42)
Scaler = StandardScaler()


# Scaling Data
x_train_Scaled = Scaler.fit_transform(x_train)
x_test_Scaled = Scaler.transform(x_test)


# Training model
C_values = [0.01, 0.1, 1, 10]
best_accuarcy = 0
Best_C_value = None
for c in C_values:
  model = LogisticRegression(C = c,max_iter= 2000, random_state = 42)
  model.fit(x_train_Scaled, y_train)
  # Prediction
  Y_pred = model.predict(x_test_Scaled)
  # Evaluation
  acc = accuracy_score(y_test, Y_pred)
  print( "C_value is :- " ,c , " Accuracy :- " , acc )

  if acc > best_accuarcy :
      best_accuarcy = acc
      Best_C_value = c
print("Best C_value is :- " ,Best_C_value)
print("Best accuracy is :- " ,best_accuarcy)


#print("Calssification report is ", classification_report(y_test, Y_pred))
#print("confusion matrix is ", confusion_matrix(y_test, Y_pred))

print("-----------------------------------------------------------------------")
print("best model")
# Train in best C_value
model = LogisticRegression(C = Best_C_value,max_iter= 2000, random_state = 42)
model.fit(x_train_Scaled, y_train)
# Prediction
Y_pred = model.predict(x_test_Scaled)
# Evaluation
print("Accuarcy is :- " ,accuracy_score(y_test, Y_pred))
print("Best C_value is :- " ,Best_C_value)
print("Best accuracy is :- " ,best_accuarcy)


# Save model  and Scaler
joblib.dump(model, 'LogisticRegression.pkl')
joblib.dump(Scaler , 'LogisticRegression_Scaler.pkl' )


















