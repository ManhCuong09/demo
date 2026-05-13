import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from supabase import create_client
from dotenv import load_dotenv

# Kiểm tra thư viện AI
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

load_dotenv()
app = Flask(__name__)

# --- Cấu hình Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Các hằng số gốc của bạn ---
LAG = 6
BET_PLAN = [1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5, 10, 11.5, 13.5, 16, 18.5, 21.5]
FREQ_FALLBACK_MIN = 3

# =========================================================
# GIỮ NGUYÊN VẸN 100% THUẬT TOÁN GỐC CỦA BẠN
# =========================================================

def fallback_distribution(digits, lag=LAG):
    probs = np.full(10, 1.0 / 10.0, dtype=float)
    if not digits:
        return probs

    total_weight = 0.0
    weighted = np.zeros(10, dtype=float)
    max_k = min(lag, len(digits) - 1)
    
    for k in range(max_k, 0, -1):
        suffix = digits[-k:]
        next_counts = np.zeros(10, dtype=float)
        matches = 0
        for i in range(k, len(digits)):
            if digits[i - k:i] == suffix:
                next_counts[digits[i]] += 1
                matches += 1
        if matches >= FREQ_FALLBACK_MIN:
            weight = float(k)
            weighted += weight * (next_counts / next_counts.sum())
            total_weight += weight

    if total_weight > 0:
        probs = weighted / total_weight
    else:
        counts = np.bincount(np.array(digits, dtype=int), minlength=10).astype(float)
        probs = counts / counts.sum()
    return probs

def train_model(digits, lag=LAG):
    if not SKLEARN_AVAILABLE:
        return None
    if len(digits) <= lag:
        return None
    
    x_rows = []
    y_rows = []
    for i in range(lag, len(digits)):
        x_rows.append(digits[i - lag:i])
        y_rows.append(digits[i])
    
    x_train, y_train = np.array(x_rows, dtype=int), np.array(y_rows, dtype=int)
    
    if len(x_train) < 20:
        return None

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "lag_digits",
                OneHotEncoder(categories=[list(range(10)) for _ in range(lag)], handle_unknown="ignore"),
                list(range(lag)),
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]
    )
    model.fit(x_train, y_train)
    return model

def predict_next_digit(digits, lag=LAG):
    if not digits:
        return {"digit": 0, "probabilities": np.full(10, 0.1), "method": "fallback_uniform"}

    if len(digits) < lag + 5:
        probs = fallback_distribution(digits, lag)
        return {"digit": int(np.argmax(probs)), "probabilities": probs, "method": "fallback_backoff"}

    model = train_model(digits, lag)
    if model is None:
        probs = fallback_distribution(digits, lag)
        return {"digit": int(np.argmax(probs)), "probabilities": probs, "method": "fallback_backoff"}

    x_next = np.array([digits[-lag:]], dtype=int)
    probs_raw = model.predict_proba(x_next)[0]
    
    full_probs = np.zeros(10, dtype=float)
    for idx, cls in enumerate(model.classes_):
        full_probs[int(cls)] = float(probs_raw[idx])
    
    if full_probs.sum() > 0:
        full_probs /= full_probs.sum()

    return {
        "digit": int(model.classes_[np.argmax(probs_raw)]),
        "probabilities": full_probs,
        "method": "logistic_regression_lag6",
    }

def get_round_info(round_no):
    round_no = max(1, min(round_no, len(BET_PLAN)))
    bet = BET_PLAN[round_no - 1]
    spent_before = sum(BET_PLAN[: round_no - 1])
    spent_after = sum(BET_PLAN[:round_no])
    net_profit_if_win = 6 * bet - spent_before
    return {
        "round": round_no,
        "bet": bet,
        "spent_before": spent_before,
        "spent_after": spent_after,
        "net_profit_if_win": net_profit_if_win,
        "rounds_left": len(BET_PLAN) - round_no,
    }

# =========================================================
# CÁC API ROUTES (ĐÃ SỬA LỖI LẤY THIẾU DỮ LIỆU)
# =========================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        # 1. Lấy 2000 số MỚI NHẤT, ép buộc sắp xếp theo ID giảm dần
        # Điều này đảm bảo ta luôn lấy được từ số 1310, 1309 lùi xuống
        res = supabase.table("results") \
            .select("id, digit") \
            .order("id", desc=True) \
            .limit(2000) \
            .execute()
        
        if not res.data:
            return jsonify({"error": "No data"}), 200

        # 2. Đảo ngược mảng để digits có thứ tự: [Số cũ -> Số mới nhất]
        # AI và thuật toán Backoff của bạn cần thứ tự này để tìm suffix
        raw_data = res.data
        digits = [item['digit'] for item in reversed(raw_data)]
        
        # 3. Chẩn đoán nhanh: Kiểm tra xem số cuối cùng có khớp với DB không
        last_id_fetched = raw_data[0]['id']
        last_digit_fetched = raw_data[0]['digit']

        # 4. Đưa vào thuật toán gốc của bạn
        prediction = predict_next_digit(digits, LAG)
        probs = prediction["probabilities"]
        top3 = sorted([(i, float(p) * 100.0) for i, p in enumerate(probs)], 
                      key=lambda x: x[1], reverse=True)[:3]

        # 5. Lấy trạng thái vòng cược
        res_state = supabase.table("app_config").select("data").eq("key", "state").execute()
        state = res_state.data[0]['data'] if res_state.data else {"current_round": 1, "stats": {}}

        return jsonify({
            "digits": digits[-80:], # Chỉ hiện 80 số cuối cho đẹp
            "debug": {
                "total_used": len(digits),
                "last_id": last_id_fetched,
                "last_digit": last_digit_fetched
            },
            "prediction": {
                "digit": prediction["digit"],
                "confidence": round(float(np.max(probs)) * 100, 2),
                "method": prediction["method"],
                "top3": " | ".join(f"{d}: {p:.2f}%" for d, p in top3)
            },
            "bet_info": get_round_info(state.get("current_round", 1)),
            "stats": state.get("stats", {"wins": 0, "busts": 0})
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/add', methods=['POST'])
def add_digit():
    val = int(request.json.get('digit'))
    last_pred = request.json.get('last_prediction')
    
    # 1. Lưu số mới vào Supabase
    supabase.table("results").insert({"digit": val}).execute()
    
    # 2. Cập nhật State (Vòng cược và Thống kê)
    res_state = supabase.table("app_config").select("data").eq("key", "state").execute()
    state = res_state.data[0]['data'] if res_state.data else {"current_round": 1, "stats": {"wins": 0, "busts": 0}}
    
    if last_pred is not None and val == int(last_pred):
        state["stats"]["wins"] = state["stats"].get("wins", 0) + 1
        state["current_round"] = 1
    else:
        if state["current_round"] >= len(BET_PLAN):
            state["stats"]["busts"] = state["stats"].get("busts", 0) + 1
            state["current_round"] = 1
        else:
            state["current_round"] += 1
            
    supabase.table("app_config").upsert({"key": "state", "data": state}).execute()
    return jsonify({"status": "ok"})

@app.route('/export-data')
def export_data():
    # SỬA LỖI: Lấy TOÀN BỘ lịch sử, không giới hạn 1000
    res = supabase.table("results").select("digit").order("id", desc=False).limit(100000).execute()
    digits = [str(item['digit']) for item in res.data]
    content = ",".join(digits)
    return Response(content, mimetype="text/plain", headers={"Content-disposition": "attachment; filename=full_data.txt"})

@app.route('/import-data-safely')
def secret_import_safely():
    # Logic nạp file giữ nguyên nhưng sửa đếm db_count chính xác
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'result.txt')
        if not os.path.exists(file_path): return "❌ Không thấy file"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        file_digits = [int(d) for d in re.findall(r'\d+', content)]
        
        res_count = supabase.table("results").select("*", count="exact").execute()
        db_count = res_count.count if res_count.count is not None else 0

        if db_count >= len(file_digits):
            return f"ℹ️ Đã đủ dữ liệu ({db_count} số)."

        new_data = file_digits[db_count:]
        rows = [{"digit": d} for d in new_data]
        
        for i in range(0, len(rows), 500):
            supabase.table("results").insert(rows[i:i+500]).execute()
            
        return f"✅ Đã nạp thêm {len(new_data)} số."
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)