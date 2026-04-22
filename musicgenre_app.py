import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib  # 모델 로드용 (또는 keras.models.load_model)

# 1. 모델 로드 (학습된 모델 파일이 필요합니다)
# model = joblib.load('music_genre_model.pkl')
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

st.title("🎵 음악 장르 분류 AI 시연")
st.write("분류하고 싶은 음악 파일(.wav)을 업로드하세요.")

uploaded_file = st.file_uploader("파일 선택", type=["wav", "mp3"])

if uploaded_file is not None:
    # 오디오 로드
    y, sr = librosa.load(uploaded_file, duration=30)
    st.audio(uploaded_file, format='audio/wav')

    # 2. 특징 시각화 (시연의 재미 요소)
    st.subheader("📊 오디오 특징 분석")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("MFCC (Mel-frequency cepstral coefficients)")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        st.pyplot(fig)

    # 3. 예측 수행 (프로젝트의 핵심 로직 연결)
    # feature = extract_features(y, sr) # 기존에 만드신 특징 추출 함수
    # prediction = model.predict(feature)
    
    st.subheader("🚀 장르 예측 결과")
    # 예시 결과 (실제 모델 예측값으로 대체하세요)
    mock_probs = [0.1, 0.05, 0.05, 0.7, 0.02, 0.01, 0.01, 0.03, 0.02, 0.01] 
    predicted_genre = genres[np.argmax(mock_probs)]
    
    st.success(f"이 곡의 장르는 **{predicted_genre.upper()}**일 확률이 높습니다!")
    st.bar_chart(dict(zip(genres, mock_probs)))
