import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 학습 때 썼던 특징들 리스트 (본인의 데이터셋 구조에 맞게 수정)
def extract_features(y, sr):
    # 특징들을 리스트로 추출
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # 각 특징의 평균값 계산 (GTZAN CSV 구조와 일치해야 함)
    feature_list = [
        np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), 
        np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)
    ]
    for e in mfcc:
        feature_list.append(np.mean(e))
        
    return np.array(feature_list).reshape(1, -1)

# 시연 부분 코드
if uploaded_file:
    y, sr = librosa.load(uploaded_file, duration=30)
    features = extract_features(y, sr)
    
    # [중요] 학습할 때 Scaler를 썼다면 여기서도 써야 합니다!
    # scaler = joblib.load('scaler.pkl') # 만약 저장해둔 스케일러가 있다면
    # features = scaler.transform(features)
    
    prediction = model.predict(features)
    # ... 결과 출력
