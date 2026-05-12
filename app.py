import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from supabase import create_client
from dotenv import load_dotenv

# Import sklearn với kiểm tra lỗi như bản gốc của bạn
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

# --- Các hằng số gốc ---
LAG = 6
BET_PLAN = [1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5, 10, 11.5, 13.5, 16, 18.5, 21.5]
FREQ_FALLBACK_MIN = 3

# --- GIỮ NGUYÊN THUẬT TOÁN GỐC CỦA BẠN ---

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
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
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

# --- CÁC API ROUTES (Kết nối Frontend và Supabase) ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    # Lấy dữ liệu từ Supabase
    res_digits = supabase.table("results").select("digit").order("id", desc=False).execute()
    digits = [item['digit'] for item in res_digits.data]
    
    # Lấy state
    res_state = supabase.table("app_config").select("data").eq("key", "state").execute()
    state = res_state.data[0]['data'] if res_state.data else {"current_round": 1, "stats": {"wins": 0, "busts": 0}}

    # Tính toán dự đoán dựa trên thuật toán gốc
    prediction = predict_next_digit(digits, LAG)
    
    # Lấy top 3
    probs = prediction["probabilities"]
    top3 = sorted([(i, float(p) * 100.0) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)[:3]
    top3_str = " | ".join(f"{d}: {p:.2f}%" for d, p in top3)

    return jsonify({
        "digits": digits[-80:], # Gửi 80 số cuối để hiển thị
        "prediction": {
            "digit": prediction["digit"],
            "confidence": round(float(np.max(probs)) * 100, 2),
            "method": prediction["method"],
            "top3": top3_str
        },
        "bet_info": get_round_info(state["current_round"]),
        "stats": state["stats"]
    })

@app.route('/api/add', methods=['POST'])
def add_digit():
    val = int(request.json.get('digit'))
    last_pred = request.json.get('last_prediction')
    
    # 1. Lưu số mới
    supabase.table("results").insert({"digit": val}).execute()
    
    # 2. Cập nhật trạng thái chu kỳ (State logic)
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
    
@app.route('/import-data-safely')
def secret_import_safely():
    try:
        # 1. Đọc file result.txt từ project
        file_path = os.path.join(os.path.dirname(__file__), 'result.txt')
        if not os.path.exists(file_path):
            return "❌ Không tìm thấy file result.txt"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. Lấy tất cả số từ file thành một danh sách
        file_digits = [int(d) for d in re.findall(r'\d+', content)]
        file_count = len(file_digits)

        if file_count == 0:
            return "⚠️ File không có dữ liệu số."

        # 3. Kiểm tra số lượng bản ghi hiện có trên Supabase
        # .count('exact') giúp lấy tổng số dòng mà không cần tải dữ liệu về
        res_count = supabase.table("results").select("*", count="exact").execute()
        db_count = res_count.count if res_count.count is not None else 0

        # 4. So sánh và xử lý
        if db_count >= file_count:
            return f"ℹ️ Không có gì mới. Database đã có {db_count} số, file có {file_count} số."

        # 5. Chỉ lấy phần số mới (phần đuôi của file mà DB chưa có)
        new_data = file_digits[db_count:]
        rows_to_insert = [{"digit": d} for d in new_data]

        # 6. Nạp phần mới vào Supabase
        chunk_size = 500
        for i in range(0, len(rows_to_insert), chunk_size):
            chunk = rows_to_insert[i:i + chunk_size]
            supabase.table("results").insert(chunk).execute()

        return f"✅ Đã nạp thêm {len(rows_to_insert)} số mới. Tổng cộng DB hiện có: {db_count + len(rows_to_insert)} số."

    except Exception as e:
        return f"❌ Lỗi hệ thống: {str(e)}"

@app.route('/export-data')
def export_data():
    try:
        # 1. Lấy toàn bộ dữ liệu từ Supabase, sắp xếp theo ID tăng dần (đúng thứ tự thời gian)
        res = supabase.table("results").select("digit").order("id", desc=False).execute()
        
        if not res.data:
            return "⚠️ Database đang trống, không có gì để xuất."

        # 2. Chuyển danh sách số thành chuỗi cách nhau bằng dấu phẩy
        digits = [str(item['digit']) for item in res.data]
        content = ",".join(digits)

        # 3. Trả về response dưới dạng file tải về
        return Response(
            content,
            mimetype="text/plain",
            headers={
                "Content-disposition": "attachment; filename=supabase_backup.txt"
            }
        )

    except Exception as e:
        return f"❌ Lỗi khi xuất dữ liệu: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)