import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# Mã hóa dữ liệu phân loại
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Sentiment"] = le.fit_transform(df["Sentiment"])
df["Purchase_Decision"] = le.fit_transform(df["Purchase_Decision"])  # Yes=1, No=0

# Biến đổi cột Brand và Fashion_Item bằng one-hot encoding
df = pd.get_dummies(df, columns=["Brand", "Fashion_Item"], drop_first=True)

# Chuẩn hóa cột số
scaler = StandardScaler()
df[["Price", "Rating", "Trendy_Score", "Age"]] = scaler.fit_transform(df[["Price", "Rating", "Trendy_Score", "Age"]])

# Chia dữ liệu cho mô hình phân loại (dự đoán quyết định mua hàng)
X_class = df.drop(columns=["Purchase_Decision", "Review_Text", "Location", "Review_Date", "User_ID"])
y_class = df["Purchase_Decision"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Chia dữ liệu cho mô hình hồi quy (dự đoán điểm xu hướng)
X_reg = df.drop(columns=["Trendy_Score", "Review_Text", "Location", "Review_Date", "User_ID"])
y_reg = df["Trendy_Score"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Huấn luyện mô hình phân loại
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
print(f"Accuracy (Purchase Decision): {acc:.2%}")

# Huấn luyện mô hình hồi quy
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)



mae = mean_absolute_error(y_test_r, y_pred_r)
print(f"Mean Absolute Error (Trendy Score): {mae:.4f}")