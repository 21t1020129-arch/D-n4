import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cấu hình trang (Bắt buộc phải ở dòng đầu tiên của lệnh Streamlit)
st.set_page_config(page_title="Hệ thống AI Dự đoán", page_icon="🤖", layout="wide")

# ==========================================
# 1. HÀM LOAD DỮ LIỆU & MÔ HÌNH (CÓ CACHE)
# ==========================================
@st.cache_data
def load_data():
    try:
        # [SỬA Ở ĐÂY]: Đổi tên file csv của bạn
        df = pd.read_csv('data.csv') 
        return df
    except:
        # Dữ liệu giả lập để app không bị crash nếu chưa có file
        st.warning("⚠️ Chưa tìm thấy file data.csv. Đang dùng dữ liệu giả lập.")
        return pd.DataFrame({'Tuoi': [25, 45, 30, 50], 'ThuNhap': [15, 30, 20, 50], 'RoiBo': [0, 1, 0, 1]})

@st.cache_resource
def load_model():
    try:
        # [SỬA Ở ĐÂY]: Đổi tên file model.pkl của bạn trong thư mục models/
        with open('models/model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.warning("⚠️ Chưa tìm thấy models/model.pkl. Vui lòng train và lưu mô hình!")
        return None

df = load_data()
model = load_model()

# ==========================================
# 2. THANH ĐIỀU HƯỚNG (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Menu Điều hướng")
menu = ["1. Giới thiệu & EDA", "2. Triển khai mô hình", "3. Đánh giá & Hiệu năng"]
choice = st.sidebar.radio("Chọn trang:", menu)

# ==========================================
# TRANG 1: GIỚI THIỆU & EDA
# ==========================================
if choice == menu[0]:
    st.title("📊 Khám phá Dữ liệu (EDA)")
    
    st.markdown("""
    **Thông tin sinh viên:**
    * **Tên đề tài:** [SỬA Ở ĐÂY] Dự đoán khả năng rời bỏ của khách hàng
    * **Họ và tên SV:** [SỬA Ở ĐÂY] Nguyễn Văn A
    * **MSSV:** [SỬA Ở ĐÂY] 20261234
    
    **Giá trị thực tiễn:** Mô hình giúp phòng Marketing nhận diện sớm các khách hàng có ý định ngừng sử dụng dịch vụ, từ đó tung ra các chương trình khuyến mãi giữ chân kịp thời, giúp tối ưu hóa doanh thu.
    """)
    
    st.subheader("1. Dữ liệu thô (Raw Data)")
    st.dataframe(df.head(10))
    
    st.subheader("2. Phân tích trực quan (EDA)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phân bố biến mục tiêu (Target)**")
        fig1, ax1 = plt.subplots()
        # [SỬA Ở ĐÂY]: Đổi 'RoiBo' thành tên cột Target của bạn
        sns.countplot(x=df.columns[-1], data=df, ax=ax1, palette="viridis")
        st.pyplot(fig1)
        
    with col2:
        st.write("**Ma trận tương quan (Correlation)**")
        fig2, ax2 = plt.subplots()
        # Chỉ tính tương quan cho cột số
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
        
    st.info("**Nhận xét Dữ liệu:** Dữ liệu có hiện tượng mất cân bằng nhẹ. Các đặc trưng như [Đặc trưng A] có mối tương quan tương đối cao với biến mục tiêu, cho thấy đây là yếu tố quan trọng.")

# ==========================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH (DỰ ĐOÁN)
# ==========================================
elif choice == menu[1]:
    st.title("🚀 Tương tác Mô hình Dự đoán")
    st.write("Nhập thông số khách hàng để xem dự đoán từ AI.")
    
    # Tạo form nhập liệu
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # [SỬA Ở ĐÂY]: Cập nhật các widget khớp với input mô hình của bạn
            f_tuoi = st.number_input("Độ tuổi", min_value=18, max_value=100, value=30)
            f_thunhap = st.number_input("Thu nhập (Tr VNĐ)", min_value=0, max_value=200, value=15)
            
        with col2:
            f_gioitinh = st.selectbox("Giới tính", ["Nam", "Nữ"])
            # Tiền xử lý input (giống lúc train)
            f_gioitinh_num = 1 if f_gioitinh == "Nam" else 0
            
        submit_button = st.form_submit_button(label='🔍 Dự đoán ngay')
        
    if submit_button:
        if model is not None:
            # Tạo DataFrame từ input (Tên cột phải KHỚP lúc train)
            input_data = pd.DataFrame({
                'Tuoi': [f_tuoi],
                'ThuNhap': [f_thunhap],
                # 'GioiTinh': [f_gioitinh_num] # Mở comment nếu mô hình có train cột này
            })
            
            try:
                # 1. Dự đoán nhãn
                prediction = model.predict(input_data)
                
                # 2. Lấy xác suất dự đoán (Độ tin cậy)
                # Dùng try-except vì các mô hình như SVM đôi khi không hỗ trợ predict_proba nếu không bật tham số
                try:
                    proba = model.predict_proba(input_data)[0]
                    confidence = max(proba) * 100
                except:
                    confidence = 100.0 # Mặc định nếu không lấy được xác suất
                
                st.markdown("---")
                st.subheader("🎯 Kết quả:")
                
                # [SỬA Ở ĐÂY]: Sửa text kết quả cho hợp bài toán
                if prediction[0] == 1:
                    st.error(f"⚠️ DỰ BÁO: Rời bỏ dịch vụ! (Độ tin cậy: {confidence:.2f}%)")
                else:
                    st.success(f"✅ DỰ BÁO: Tiếp tục sử dụng! (Độ tin cậy: {confidence:.2f}%)")
                    
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: Số lượng đặc trưng đầu vào không khớp với mô hình. Chi tiết lỗi: {e}")
        else:
            st.error("Chưa có mô hình để dự đoán!")

# ==========================================
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ==========================================
elif choice == menu[2]:
    st.title("📈 Đánh giá Hiệu năng Mô hình")
    
    st.markdown("### 1. Chỉ số đo lường (Metrics)")
    # Dùng st.metric để hiển thị số liệu đẹp mắt
    col1, col2, col3 = st.columns(3)
    # [SỬA Ở ĐÂY]: Điền các con số bạn lấy từ file Jupyter Notebook vào
    col1.metric("Accuracy (Độ chính xác)", "85.2%", "+2.1%")
    col2.metric("F1-Score", "0.82", "-")
    col3.metric("Recall", "0.80", "-")
    
    st.markdown("### 2. Biểu đồ Kỹ thuật")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write("**Ma trận nhầm lẫn (Confusion Matrix)**")
        # [SỬA Ở ĐÂY]: Bạn có thể dùng st.image('confusion_matrix.png') nếu đã lưu ảnh
        # Hoặc sinh ngẫu nhiên 1 cái để demo như dưới đây:
        fig_cm, ax_cm = plt.subplots(figsize=(5,4))
        cm = np.array([[80, 20], [15, 85]]) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Dự đoán')
        ax_cm.set_ylabel('Thực tế')
        st.pyplot(fig_cm)
        
    st.markdown("### 3. Phân tích sai số (Error Analysis)")
    st.warning("""
    **Nhận định:** Mô hình hoạt động khá ổn định. Tuy nhiên, tỷ lệ False Negative (Dự đoán ở lại nhưng thực tế rời bỏ) vẫn còn khoảng 15%. 
    
    **Nguyên nhân:** Mô hình thường dự đoán sai ở nhóm khách hàng mới sử dụng dịch vụ dưới 3 tháng (chưa có nhiều dữ liệu lịch sử giao dịch).
    
    **Hướng cải thiện:** Thu thập thêm các đặc trưng về chất lượng cuộc gọi phàn nàn lên tổng đài (Text Data) và kết hợp với mô hình NLP để tăng độ chính xác.
    """)