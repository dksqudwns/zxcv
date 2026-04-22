import streamlit as st
import librosa
import numpy as np
import joblib
import os

# 1. 모델과 스케일러 파일이 있는지 확인 후 로드
model_path = 'music_genre_model.pkl'
scaler_path = 'scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.error("모델 파일(.pkl)이나 스케일러 파일(scaler.pkl)이 GitHub 저장소에 없습니다!")

def extract_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    feature_list = [
        np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), 
        np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
    ]
    for e in mfcc:
        feature_list.append(np.mean(e))
        
    return np.array(feature_list).reshape(1, -1)

# --- UI 부분 ---
st.title("🎵 음악 장르 분류 AI 시연")

# 2. 변수 정의 (이게 위쪽에 있어야 NameError가 안 납니다)
uploaded_file = st.file_uploader("음악 파일을 업로드하세요.", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('분석 중...'):
        y, sr = librosa.load(uploaded_file, duration=30)
        
        # 특징 추출 및 스케일링
        features = extract_features(y, sr)
        
        try:
            # 학습 때 썼던 그 스케일러로 데이터를 보정해야 결과가 정확합니다.
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            result = genres[np.argmax(prediction)]
            
            st.success(f"예측 결과: **{result.upper()}**")
            st.bar_chart(dict(zip(genres, prediction[0].tolist())))
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
