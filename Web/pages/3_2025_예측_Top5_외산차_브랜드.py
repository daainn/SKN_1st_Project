from modules.data_select import get_domestic_data, get_brand_registration_data, create_connection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

## -------------matplotlib font설정 및 테마 설정----------------
sns.set_theme(style="whitegrid")

# 격자선 색을 연한 색으로 설정
plt.rcParams["grid.color"] = "#fff0f5"
font_path = "C:\\Users\\Playdata\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Pretendard-Regular.otf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)


# ------------------------MySQL 연결--------------------

@st.cache_resource
def get_connection():
    return create_connection()

@st.cache_data
def load_brand_registration_data():
    connection = get_connection()  # get_connection()에서 캐시된 연결을 가져옵니다.
    return get_brand_registration_data(connection)

def get_connection():
    return create_connection()

def load_brand_registration_data():
    connection = get_connection()  # get_connection()에서 새로 연결을 가져옵니다.
    return get_brand_registration_data(connection)


def initialize_session_data():
    if "brand_registration_data" not in st.session_state:
        st.session_state.brand_registration_data = load_brand_registration_data()

initialize_session_data()


## ---------------------매인 텍스트 영역 -----------------------

df = st.session_state.brand_registration_data
df = df[df['BrandName'] != "Total"]  # Total 행 제거
# 상위 5개 브랜드 추출을 위한 데이터프레임 복사
df = df.copy()

# 2025년 예측을 위한 선형회귀 수행
future_year = 2025
predictions = []

for brand in df['BrandName'].unique():  # 브랜드별로 처리
    # 해당 브랜드의 데이터 필터링
    brand_data = df[df['BrandName'] == brand]
    
    # 독립 변수와 종속 변수 정의 (년도와 점유율)
    x = brand_data['YearID'].values.reshape(-1, 1)  # 년도
    y = brand_data['MarketShare'].values           # 점유율
    
    # 선형회귀 모델 훈련
    model = LinearRegression()
    model.fit(x, y)
    
    # 2025년 점유율 예측
    pred_2025 = model.predict([[future_year]])[0]
    predictions.append((brand, pred_2025))

# 예측 결과를 데이터프레임으로 변환
prediction_df = pd.DataFrame(predictions, columns=["BrandName", "2025 MarketShare"])
prediction_df = prediction_df.sort_values(by="2025 MarketShare", ascending=False)

# 상위 5개 브랜드 추출
top5_brands = prediction_df.head(5)
#---------------------------------제목------------
# Streamlit 앱 UI 구성

st.markdown("""
    <div style="background-color: #fffafa; padding: 10px; border-radius: 3px; margin-bottom: 10px;">
        <h1 style="color: black; font-size: 3x; text-align: center;">📈2025 예측 Top5 외산차 브랜드</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <hr style="border: none; border-top: 0.1px solid #454876; margin-top: 20px; margin-bottom: 40px;">
""", unsafe_allow_html=True)

# ---------------------테이블---------------
# 구분선


st.write("과거 11개년의 데이터를 기반으로 2025 브랜드별 점유율을 예측")
st.subheader("2025 외산차 브랜드 점유율")

# 예측 결과 데이터프레임 출력
st.dataframe(prediction_df, use_container_width=True)
# 상위 5개 브랜드 시각화
st.markdown("""
    <hr style="border: none; border-top: 0.1px solid #454876; margin-top: 20px; margin-bottom: 40px;">
""", unsafe_allow_html=True)
# ---------------------그래프프---------------
# 그래프 그리기
st.subheader("2025 Top 5 외산차 브랜드")
colors = sns.color_palette("Pastel1", len(top5_brands))  # 색상 정의
plt.figure(figsize=(10, 6))

for idx, (brand_name, market_share) in enumerate(zip(top5_brands["BrandName"], top5_brands["2025 MarketShare"])):
    plt.bar(brand_name, market_share, color=colors[idx % len(colors)],  # 색상 참조
            label=f"{brand_name}: {market_share:.2f}%")

plt.xlabel("Brand Name")
plt.ylabel("Predicted Market Share (%)")
plt.title("Top 5 Predicted Market Share in 2025")
plt.legend()
st.pyplot(plt)

st.markdown("""
    <hr style="border: none; border-top: 0.1px solid #454876; margin-top: 20px; margin-bottom: 40px;">
""", unsafe_allow_html=True)
#-----------------그래프 그리기---------------------------
# 데이터 준비
# 데이터 준비

# 기존 데이터에 2025년 예측 데이터 결합
for brand in prediction_df['BrandName']:
    pred_value = prediction_df[prediction_df['BrandName'] == brand]["2025 MarketShare"].values[0]
    df.loc[df['BrandName'] == brand, 'MarketShare_2025'] = pred_value

# 2014년부터 2025년까지 년도별 데이터 시각화
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

for brand in df['BrandName'].unique():
    # 해당 브랜드의 데이터 필터링
    brand_data = df[df['BrandName'] == brand]
    
    # 2014년부터 2025년까지의 년도와 점유율 추출
    years = list(range(2014, 2026))
    market_share_values = list(brand_data[brand_data['YearID'] <= 2025].sort_values('YearID')['MarketShare']) + [prediction_df[prediction_df['BrandName'] == brand]["2025 MarketShare"].values[0]]

    # 선형 그래프 그리기
    plt.plot(years, market_share_values, label=brand, marker='o', linestyle='-', linewidth=2)

plt.xlabel("Year")
plt.ylabel("Market Share (%)")
plt.title("Brand-wise Market Share Prediction from 2014 to 2025")
plt.legend(title="Brand", loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
plt.tight_layout()

# Streamlit에 그래프 출력
st.subheader("2014 - 2025년까지의 외산차 브랜드 점유율")
st.pyplot(plt)