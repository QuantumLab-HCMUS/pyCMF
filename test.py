import pycmf
from pyscf import gto, scf

print("=======================================================")
print("PHẦN 1: KIỂM TRA MẶT TIỀN THƯ VIỆN (SMOKE TEST)")
print("=======================================================")

# Danh sách 21 hàm bọc được trích xuất từ __init__.py
danh_sach_ham = [
    # Hệ kín
    "OBMP2", "OBMP2_faster", "OBMP2_active", "OBMP2_mod", "KOBMP2",
    "DFOBMP2_faster_ram", "DFOBMP2_slower", "DFTOBMP2",
    # Hệ mở
    "UOBMP2", "UOBMP2_faster", "UOBMP2_SCS", "UOBMP2_MOM", "UOBMP2_mom_conv",
    "UOBMP2_dfold", "UOBMP2_active", "UOBMP2_active_scf", "DFUOBMP2_ram_reduced",
    "DFUOBMP2_ram_reduced_new", "DFUOBMP2_mom_conv", "DFUOBMP2_faster_ram", "DFTUOBMP2"
]

loi_import = 0
for ten_ham in danh_sach_ham:
    if hasattr(pycmf, ten_ham):
        print(f"  ✅ Đã tìm thấy: {ten_ham}")
    else:
        print(f"  ❌ LỖI: Không tìm thấy {ten_ham}")
        loi_import += 1

if loi_import > 0:
    print(f"\n⚠️ Phát hiện {loi_import} lỗi import. Hãy dừng lại và kiểm tra mã nguồn.")
    exit(1)
else:
    print("\n🎉 Tuyệt vời! Cả 21 thuật toán đều đã sẵn sàng.")

print("\n=======================================================")
print("PHẦN 2: CHẠY THỬ NGHIỆM TÍNH TOÁN THỰC TẾ (FUNCTIONAL TEST)")
print("=======================================================")

# 1. Khởi tạo phân tử Nước (H2O) đơn giản
print("Đang cấu hình phân tử H2O (basis: sto-3g)...")
mol = gto.M(
    atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
    basis='sto-3g',
    verbose=0 # Tắt log của PySCF cho dễ nhìn
)

# 2. Test Hệ Kín (RHF)
print("\n--- Chạy thử Hệ Kín (RHF) với thuật toán OBMP2 ---")
mf_rhf = scf.RHF(mol).run()
try:
    # Gọi hàm OBMP2 từ thư viện pycmf
    my_obmp2 = pycmf.OBMP2(mf_rhf)
    print("  ✅ Khởi tạo class OBMP2 thành công!")
    # Tùy thuộc vào việc bạn đã code hàm kernel() chưa, bạn có thể mở comment dòng dưới để chạy:
    # e_corr = my_obmp2.kernel()[0]
    # print(f"  ✅ Năng lượng tương quan tính được: {e_corr}")
except Exception as e:
    print(f"  ❌ LỖI khi chạy OBMP2: {e}")

# 3. Test Hệ Mở (UHF)
print("\n--- Chạy thử Hệ Mở (UHF) với thuật toán UOBMP2 ---")
mf_uhf = scf.UHF(mol).run()
try:
    # Gọi hàm UOBMP2 từ thư viện pycmf
    my_uobmp2 = pycmf.UOBMP2(mf_uhf)
    print("  ✅ Khởi tạo class UOBMP2 thành công!")
except Exception as e:
    print(f"  ❌ LỖI khi chạy UOBMP2: {e}")

print("\n=======================================================")
print("HOÀN TẤT KIỂM TRA!")
print("=======================================================")