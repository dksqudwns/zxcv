import streamlit as st
import librosa
import numpy as np
import joblib

# 1. 모델과 스케일러 로드 (파일명이 본인 파일과 맞는지 확인!)
# 저장소에 모델과 스케일러 파일이 있어야 합니다.
model = joblib.load('music_genre_model.pkl') 
scaler = joblib.load('scaler.pkl') # 학습 시 사용했던 스케일러 파일

def extract_features(y, sr):
    # 특징 추출 (사용하신 데이터셋 구조와 동일해야 함)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # 평균값 계산
    feature_list = [
        np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), 
        np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
    ]
    for e in mfcc:
        feature_list.append(np.mean(e))
        
    return np.array(feature_list).reshape(1, -1)

# --- 시연 UI 부분 ---
st.title("🎵 음악 장르 분류 AI 시연")

# 에러 해결을 위해 변수를 정확히 선언
uploaded_file = st.file_uploader("음악 파일(.wav, .mp3)을 업로드하세요.", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('분석 중...'):
        # 오디오 로드 및 무음 제거 (정확도 향상)
        y, sr = librosa.load(uploaded_file, duration=30)
        y, _ = librosa.effects.trim(y) 
        
        # 특징 추출
        features = extract_features(y, sr)
        
        # [핵심] 스케일링 적용 (이게 없으면 다 똑같이 나옵니다)
        scaled_features = scaler.transform(features)
        
        # 예측
        prediction = model.predict(scaled_features)
        
        # 결과 출력 (장르 리스트는 본인 순서에 맞게 수정)
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        result = genres[np.argmax(prediction)]
        
        st.success(f"예측 결과: **{result.upper()}**")
        st.bar_chart(dict(zip(genres, prediction[0])))
