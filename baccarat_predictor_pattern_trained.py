
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ฝึกโมเดลจาก pattern จำลอง (มังกร, ปิงปอง, คู่)
@st.cache_resource
def train_model():
    import pandas as pd
    import random
    def generate_patterned_baccarat(rounds=10000):
        results = []
        patterns = ['dragon', 'pingpong', 'pair', 'random']
        for _ in range(rounds // 10):
            pattern = random.choice(patterns)
            if pattern == 'dragon':
                side = random.choice(['P', 'B'])
                results.extend([side] * 7 + [random.choice(['P', 'B'])] * 3)
            elif pattern == 'pingpong':
                base = random.choice([('P', 'B'), ('B', 'P')])
                for i in range(10):
                    results.append(base[i % 2])
            elif pattern == 'pair':
                base = random.choice(['P', 'B'])
                pair = [base, base]
                results.extend(pair * 5)
            else:
                results.extend(random.choices(['P', 'B', 'T'], weights=[0.45, 0.45, 0.1], k=10))
        return results

    simulated_results = generate_patterned_baccarat()
    df = pd.DataFrame({'result': simulated_results})
    df['result_code'] = df['result'].map({'P': 0, 'B': 1, 'T': 2})

    for i in range(1, 4):
        df[f'prev_{i}'] = df['result_code'].shift(i)
    df.dropna(inplace=True)

    X = df[['prev_1', 'prev_2', 'prev_3']]
    y = df['result_code']

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Mapping
label_map = {'P': 0, 'B': 1, 'T': 2}
reverse_map = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}

# UI
st.title("Baccarat Predictor - Pattern Trained 3-Turns Pro")
st.write("พิมพ์ผล 3 ตาหลังสุด เช่น: `B P B` หรือ `P P T`")

input_text = st.text_input("ผล 3 ตาหลังสุด (คั่นด้วยเว้นวรรค)").strip().upper()

if input_text:
    tokens = input_text.split()
    if len(tokens) != 3 or any(t not in label_map for t in tokens):
        st.error("กรุณาพิมพ์ให้ถูกต้อง เช่น: `B P B` หรือ `P T P`")
    else:
        code_seq = [label_map[t] for t in tokens]
        pred = model.predict([code_seq])[0]
        st.success(f"ระบบคาดการณ์ว่า: **{reverse_map[pred]}**")
