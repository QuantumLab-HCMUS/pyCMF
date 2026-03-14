import os
import subprocess

# ==========================================
# CẤU HÌNH THƯ MỤC VÀ FILE CHO DỰ ÁN PYCMF
# ==========================================

# Khai báo đường dẫn gốc chứa source code
BASE_DIR = "src/pycmf"

# 1. Danh sách các thư mục mới cần tạo
NEW_DIRS = [
    os.path.join(BASE_DIR, "OBMP"),
    os.path.join(BASE_DIR, "OBDF"),
    os.path.join(BASE_DIR, "OBDH"),
    os.path.join(BASE_DIR, "KOBMP")
]

# 2. Bảng quy hoạch file: "Đường_dẫn_cũ" : "Đường_dẫn_mới"
FILE_MAPPING_RAW = {
    # Từ nhánh Restricted (obmp2)
    "obmp2/obmp2.py": "OBMP/obmp2_slow.py",
    "obmp2/kobmp2.py": "KOBMP/kobmp2.py",
    "obmp2/obmp2_active.py": "OBMP/obmp2_cas.py",
    "obmp2/dftobmp2.py": "OBDH/dft_obmp2.py",
    "obmp2/dfobmp2_faster_ram.py": "OBDF/dfobmp2.py",
    "obmp2/dfobmp2_slower.py": "OBDF/dfobmp2_slow.py",
    "obmp2/obmp2_faster.py": "OBMP/obmp2.py",
    "obmp2/obmp2mod.py": "OBMP/obmp2_einsum.py",

    # Từ nhánh Unrestricted (uobmp2)
    "uobmp2/uobmp2.py": "OBMP/uobmp2_slow.py",
    "uobmp2/uobmp2_scs.py": "OBMP/uobmp2_scs.py",
    "uobmp2/uobmp2_mom.py": "OBMP/uobmp2_mom.py",
    "uobmp2/uobmp2_mom_conv.py": "OBMP/uobmp2_mom_diis.py",
    "uobmp2/uobmp2_faster.py": "OBMP/uobmp2.py",
    "uobmp2/dftuobmp2.py": "OBDH/dft_uobmp2.py",
    "uobmp2/dfuobmp2_faster_ram.py": "OBDF/dfuobmp2_mom.py",
    "uobmp2/dfuobmp2_mom_conv.py": "OBDF/dfuobmp2_mom_diis.py",
    "uobmp2/dfuobmp2_ram_reduced.py": "OBDF/dfuobmp2_einsum.py",
    "uobmp2/dfuobmp2_ram_reduced_new.py": "OBDF/dfuobmp2.py",
    
    # BỔ SUNG 3 FILE BỊ SÓT
    "uobmp2/uobmp2_active.py": "OBMP/uobmp2_cas.py",
    "uobmp2/uobmp2_active_scf.py": "OBMP/uobmp2_cas_scf.py",
    "uobmp2/uobmp2_dfold.py": "OBMP/uobmp2_downfold.py"
}

# Tự động gộp đường dẫn gốc vào file mapping để Git có thể tìm thấy
FILE_MAPPING = {os.path.join(BASE_DIR, k): os.path.join(BASE_DIR, v) for k, v in FILE_MAPPING_RAW.items()}

# 3. Từ điển sửa lỗi Import tự động
IMPORT_REPLACEMENTS = {
    "from pycmf.obmp2 import obmp2, dfobmp2_faster_ram, dfobmp2_slower": "from pycmf.OBMP import obmp2_slow as obmp2\nfrom pycmf.OBDF import dfobmp2, dfobmp2_slow",
    "from pycmf.uobmp2 import uobmp2_mom_conv, dfuobmp2_ram_reduced": "from pycmf.OBMP import uobmp2_mom_diis\nfrom pycmf.OBDF import dfuobmp2_einsum",
    "from pycmf.obmp2 import dfobmp2_faster_ram, obmp2": "from pycmf.OBDF import dfobmp2\nfrom pycmf.OBMP import obmp2_slow as obmp2",
    "from pycmf.obmp2 import obmp2, dfobmp2_faster_ram": "from pycmf.OBMP import obmp2_slow as obmp2\nfrom pycmf.OBDF import dfobmp2",
    "from pycmf.obmp2 import obmp2_faster": "from pycmf.OBMP import obmp2",
    "from pycmf.obmp2 import obmp2": "from pycmf.OBMP import obmp2_slow as obmp2",
    "from pycmf.obmp2 import dfobmp2_faster_ram": "from pycmf.OBDF import dfobmp2",
    "from . import obmp2_faster": "from ..OBMP import obmp2",
    "from . import dfobmp2_faster_ram": "from ..OBDF import dfobmp2",
    "from ..obmp2 import OBMP2, _ChemistsERIs, DFOBMP2": "from ..OBMP.obmp2 import OBMP2, _ChemistsERIs\nfrom ..OBDF.dfobmp2 import DFOBMP2",
    "from ..obmp2 import OBMP2, _ChemistsERIs": "from ..OBMP.obmp2 import OBMP2, _ChemistsERIs",
}

# ==========================================
# BẮT ĐẦU THỰC THI (AN TOÀN)
# ==========================================

def run_refactor():
    print("🚀 Bắt đầu quá trình Tái cấu trúc thư viện pyCMF...\n")

    # Bước 1: Tạo thư mục mới và file __init__.py
    for folder in NEW_DIRS:
        os.makedirs(folder, exist_ok=True)
        init_file = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# Khởi tạo package {os.path.basename(folder)}\n")
        print(f"✅ Đã chuẩn bị thư mục: {folder}/")

    print("\n📦 Bắt đầu chuyển và đổi tên file bằng Git...")
    # Bước 2: Dùng lệnh Git để di chuyển và đổi tên
    for old_path, new_path in FILE_MAPPING.items():
        if os.path.exists(old_path):
            result = subprocess.run(["git", "mv", old_path, new_path], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   🔄 Đã chuyển: {old_path} -> {new_path}")
            else:
                print(f"   ❌ Lỗi Git khi chuyển {old_path}: {result.stderr.strip()}")
        else:
            print(f"   ⚠️ Bỏ qua (không tìm thấy): {old_path}")

    print("\n🔍 Bắt đầu cập nhật lại các lệnh Import trong file...")
    # Bước 3: Quét và sửa các lệnh import
    for new_path in FILE_MAPPING.values():
        if os.path.exists(new_path):
            with open(new_path, 'r', encoding='utf-8') as file:
                code_content = file.read()

            updated = False
            for old_import, new_import in IMPORT_REPLACEMENTS.items():
                if old_import in code_content:
                    code_content = code_content.replace(old_import, new_import)
                    updated = True

            if updated:
                with open(new_path, 'w', encoding='utf-8') as file:
                    file.write(code_content)
                print(f"   🛠️  Đã sửa import thành công trong: {new_path}")

    print("\n🎉 HOÀN TẤT! Kiến trúc pyCMF đã được nâng cấp lên chuẩn chuyên gia!")

if __name__ == "__main__":
    run_refactor()