import os
import re
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path

# 1. Load cấu hình từ file .env
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

def import_data_from_txt(file_name):
    # Xác định đường dẫn file
    file_path = Path(__file__).parent / file_name
    
    if not file_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file {file_name}")
        return

    # 2. Đọc nội dung file
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # 3. Dùng Regex để tìm tất cả các chữ số (bất kể định dạng dấu phẩy hay xuống dòng)
        # Cách này an toàn hơn split(",") thông thường
        digits = re.findall(r'\d+', content)
        
        if not digits:
            print("⚠️ File trống hoặc không có dữ liệu số.")
            return

        # Chuyển sang định dạng danh sách dictionary để insert hàng loạt
        rows = [{"digit": int(d)} for d in digits]
        
        print(f"🔄 Đang đọc {len(rows)} số từ file {file_name}...")

        # 4. Chia nhỏ để insert (nếu dữ liệu quá lớn, ví dụ > 1000 số)
        # Supabase/Postgres có giới hạn số lượng dòng trong 1 lần insert tùy cấu hình
        chunk_size = 500
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            supabase.table("results").insert(chunk).execute()
            print(f"✅ Đã nạp xong {i + len(chunk)}/{len(rows)} số.")

        print("🚀 Hoàn thành! Dữ liệu đã sẵn sàng trên Cloud.")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Thay 'result.txt' bằng tên file thật của bạn
    import_data_from_txt("result.txt")