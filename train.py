import pickle
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# 1. Tạo dữ liệu giả lập (Khớp với 2 ô nhập liệu Tuoi và ThuNhap trên web)
data = {
    'Tuoi': [25, 45, 30, 50, 22, 60, 35, 28],
    'ThuNhap': [15, 30, 20, 50, 10, 80, 25, 18],
    'RoiBo': [0, 1, 0, 1, 0, 1, 0, 0] # 1 là rời bỏ, 0 là ở lại
}
df = pd.DataFrame(data)

# Tách đặc trưng (X) và nhãn (y)
X = df[['Tuoi', 'ThuNhap']]
y = df['RoiBo']

# 2. Huấn luyện mô hình Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 3. Đảm bảo thư mục models/ tồn tại và lưu mô hình
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ ĐÃ LƯU THÀNH CÔNG: models/model.pkl. Hãy quay lại web và tải lại trang!")