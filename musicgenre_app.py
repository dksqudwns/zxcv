import streamlit as st
import librosa
import numpy as np
import joblib
import os

# 1. 모델과 스케일러 파일이 있는지 확인 후 로드
# GitHub에 실제로 올린 파일 이름과 똑같이 맞춥니다.
model_path = 'model.pkl'
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
            # 1. 데이터를 보정하고 확률을 예측합니다.
            scaled_features = scaler.transform(features)
            
            # 중요! .predict() 대신 .predict_proba()를 사용해야 그래프가 예쁘게 나옵니다.
            prediction_proba = model.predict_proba(scaled_features)[0]
            
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            
            # 가장 높은 확률을 가진 장르를 뽑습니다.
            result = genres[np.argmax(prediction_proba)]
            
            # 2. 결과 출력
            st.success(f"예측 결과: **{result.upper()}**")
            
            # 3. 그래프 깔끔하게 그리기
            import pandas as pd
            chart_data = pd.DataFrame({
                'Genre': genres,
                'Probability': prediction_proba
            })
            
            # 확률이 높은 순서대로 정렬해서 가로로 보여줍니다.
            chart_data = chart_data.sort_values(by='Probability', ascending=True)
            st.bar_chart(chart_data.set_index('Genre'))

        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
