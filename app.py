import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import datetime

# ======================================================
# 1. CẤU HÌNH & API
# ======================================================
TMDB_API_KEY = "973eac1c6ee5c0af02fd6281ff2bb30b" # Key của bạn

st.set_page_config(page_title="Hệ thống Gợi ý Phim", layout="wide")

# Khởi tạo Session State để lưu lịch sử (Bộ nhớ tạm thời)
if 'history' not in st.session_state:
    st.session_state['history'] = []

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

# ======================================================
# 2. XỬ LÝ DỮ LIỆU & AI
# ======================================================
@st.cache_data
def load_data_and_model():
    df = pd.read_csv('movies_clean.csv')
    # Tạo soup vector
    df['soup'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'].fillna(''))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

with st.spinner('Đang khởi động hệ thống AI...'):
    df, cosine_sim = load_data_and_model()

# ======================================================
# 3. SIDEBAR - LỊCH SỬ NGƯỜI DÙNG (Tính năng Nâng Cao)
# ======================================================
st.sidebar.title("👤 Hồ sơ người dùng")
st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Lịch sử tìm kiếm")

# Hiển thị lịch sử từ mới nhất đến cũ nhất
if len(st.session_state['history']) > 0:
    for item in reversed(st.session_state['history']):
        st.sidebar.text(f"• {item}")
    
    if st.sidebar.button("Xóa lịch sử"):
        st.session_state['history'] = []
        st.rerun() # Load lại trang
else:
    st.sidebar.info("Chưa có hoạt động nào.")

# ======================================================
# 4. GIAO DIỆN CHÍNH
# ======================================================
st.title("🎬 Movie Recommender System")
st.markdown("### Đồ án Final Project - AI Engineer")

# Tabs: Chia giao diện thành 2 phần
tab1, tab2 = st.tabs(["🔍 Gợi ý Phim", "📊 Phân tích dữ liệu"])

with tab1:
    movie_list = df['original_title'].values
    selected_movie = st.selectbox("Bạn thích phim nào?", movie_list)

    if st.button('🚀 Gợi ý cho tôi'):
        # 1. Lưu vào lịch sử
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{selected_movie} ({timestamp})"
        st.session_state['history'].append(log_entry)
        
        # 2. Xử lý gợi ý
        st.write(f"Những bộ phim tương tự với **{selected_movie}**:")
        
        # Logic gợi ý (như cũ)
        indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()
        if selected_movie in indices:
            idx = indices[selected_movie]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]
            
            cols = st.columns(5)
            for i, col in enumerate(cols):
                movie_idx = sim_scores[i][0]
                title = df.iloc[movie_idx].original_title
                movie_id = df.iloc[movie_idx].id
                poster = fetch_poster(movie_id)
                
                with col:
                    st.image(poster)
                    st.caption(title)
        else:
            st.error("Không tìm thấy phim này trong dữ liệu!")

with tab2:
    st.header("📊 Phân tích dữ liệu (EDA)")
    st.write("Thống kê tổng quan về bộ dữ liệu phim TMDB 5000:")
    
    # Hiển thị số liệu tổng quan (KPIs)
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số phim", df.shape[0])
    col2.metric("Số lượng từ khóa", df['keywords'].nunique()) # Ví dụ minh họa
    col3.metric("Điểm đánh giá TB", round(df['vote_average'].mean(), 2))
    
    st.markdown("---")

    # Hiển thị biểu đồ 1 và 2 song song
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Thể loại phổ biến")
        try:
            st.image("chart_top_genres.png", use_container_width=True)
        except:
            st.error("Chưa thấy file ảnh. Hãy chạy lại step2_cleaning_eda.py")
            
    with c2:
        st.subheader("Phân bố điểm đánh giá")
        try:
            st.image("chart_rating_distribution.png", use_container_width=True)
        except:
            st.error("Chưa thấy file ảnh.")

    st.markdown("---")
    
    # Hiển thị WordCloud lớn ở dưới
    st.subheader("☁️ WordCloud: Các từ khóa nổi bật")
    try:
        st.image("chart_wordcloud.png", use_container_width=True)
    except:
        st.write("Chưa có ảnh WordCloud")