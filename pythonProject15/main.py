# ====== 1. Data Collection ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Giữ kết quả random cố định
np.random.seed(42)

# Tạo dữ liệu giả lập
months = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
products = [f"Product {i+1}" for i in range(200)]  # 200 sản phẩm

data = []
for product in products:
    for date in months:
        sales = np.random.randint(150, 500)        # Số lượng bán
        price = np.random.uniform(20.0, 30.0)      # Giá mỗi sản phẩm
        revenue = round(sales * price, 2)          # Doanh thu
        data.append([date, product, sales, round(price, 2), revenue])

df = pd.DataFrame(data, columns=["Date", "Product", "Sales", "Unit_Price", "Revenue"])

# ====== 2. Data Preprocessing ======
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Kiểm tra dữ liệu thiếu
print("Missing values:\n", df.isnull().sum())

# ====== 3. EDA (Exploratory Data Analysis) ======
monthly_revenue = df.groupby("Month")["Revenue"].sum()
print("\nTotal revenue by month:\n", monthly_revenue)

# Vẽ biểu đồ doanh thu theo tháng
plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_revenue.index, y=monthly_revenue.values, palette='viridis')
plt.title("Total Revenue by Month", fontsize=14)
plt.ylabel("Revenue (USD)")
plt.xlabel("Month")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ====== 4. ML Model (Dự đoán doanh thu tháng tiếp theo) ======
X = monthly_revenue.index.values.reshape(-1, 1)  # Tháng (1-12)
y = monthly_revenue.values                       # Doanh thu

model = LinearRegression()
model.fit(X, y)

pred_month = 13
predicted_revenue = model.predict([[pred_month]])
print(f"\nDự đoán doanh thu cho tháng {pred_month}: {predicted_revenue[0]:.2f} USD")

# ====== 5. Asset (Xuất dữ liệu) ======
df.to_csv("sales_data_200_products.csv", index=False)
print("\nFile sales_data_200_products.csv đã được lưu.")
print("Đường dẫn file:", os.path.abspath("sales_data_200_products.csv"))