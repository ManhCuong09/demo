import os
from flask import Flask, render_template, request, jsonify
from supabase import create_client
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- Cấu hình Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Logic hằng số ---
LAG = 6
BET_PLAN = [1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5, 10, 11.5, 13.5, 16, 18.5, 21.5]

# --- Các hàm logic từ code cũ của bạn ---
def get_round_info(round_no):
    round_no = max(1, min(round_no, len(BET_PLAN)))
    bet = BET_PLAN[round_no - 1]
    spent_before = sum(BET_PLAN[: round_no - 1])
    return {
        "round": round_no,
        "bet": bet,
        "profit": round(6 * bet - spent_before, 2),
        "left": len(BET_PLAN) - round_no
    }

def train_and_predict(digits):
    if len(digits) < LAG + 5:
        # Fallback đơn giản nếu thiếu dữ liệu
        counts = np.bincount(digits, minlength=10) if digits else np.ones(10)
        probs = counts / counts.sum()
        return int(np.argmax(probs)), float(np.max(probs)), "fallback_frequency"

    x_train = [digits[i-LAG:i] for i in range(LAG, len(digits))]
    y_train = digits[LAG:]
    
    preprocessor = ColumnTransformer([("lag", OneHotEncoder(categories=[list(range(10))]*LAG), list(range(LAG)))])
    model = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    
    model.fit(x_train, y_train)
    x_next = np.array([digits[-LAG:]])
    probs = model.predict_proba(x_next)[0]
    pred = int(model.classes_[np.argmax(probs)])
    return pred, float(np.max(probs)), "logistic_regression_lag6"

# --- API Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    # Lấy 100 số gần nhất từ Supabase
    res_digits = supabase.table("results").select("digit").order("id", desc=True).limit(100).execute()
    digits = [item['digit'] for item in reversed(res_digits.data)]
    
    # Lấy trạng thái (state)
    res_state = supabase.table("app_config").select("data").eq("key", "state").execute()
    state = res_state.data[0]['data'] if res_state.data else {"current_round": 1, "stats": {"wins": 0, "busts": 0}}

    pred_digit, confidence, method = train_and_predict(digits)
    
    return jsonify({
        "digits": digits,
        "prediction": {"digit": pred_digit, "confidence": round(confidence * 100, 2), "method": method},
        "bet_info": get_round_info(state["current_round"]),
        "stats": state["stats"]
    })

@app.route('/api/add', methods=['POST'])
def add_digit():
    val = int(request.json.get('digit'))
    # 1. Lưu số mới vào Supabase
    supabase.table("results").insert({"digit": val}).execute()
    
    # 2. Xử lý logic vòng cược (state)
    res_state = supabase.table("app_config").select("data").eq("key", "state").execute()
    state = res_state.data[0]['data'] if res_state.data else {"current_round": 1, "stats": {"wins": 0, "busts": 0}}
    
    # Ở đây bạn cần logic so sánh với 'last_prediction' để biết thắng hay thua
    # Để đơn giản, giả sử frontend gửi kèm kết quả dự đoán trước đó
    last_pred = request.json.get('last_prediction')
    
    if last_pred is not None and val == int(last_pred):
        state["stats"]["wins"] += 1
        state["current_round"] = 1
    else:
        if state["current_round"] >= len(BET_PLAN):
            state["stats"]["busts"] += 1
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