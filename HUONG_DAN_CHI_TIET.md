# HƯỚNG DẪN CHI TIẾT TỪ NGUYÊN LÝ GỐC (FIRST PRINCIPLES)
## LAB DAY 10: FINETUNE QWEN3-VL-4B (LoRA) ĐỊNH VỊ HƯ HẠI MẶT ĐƯỜNG

> **Phương pháp tiếp cận:** Phân tích từ nguyên lý gốc (First Principles Thinking), bóc tách mọi cơ chế đến tầng bản chất vật lý / toán học, đi kèm ví dụ đời thường sinh động và mã thực thi cụ thể từng bước.

---

## MỤC LỤC

1. [Phần 1: Tư Duy Nguyên Lý Gốc – Bản Chất Vấn Đề](#phần-1-tư-duy-nguyên-lý-gốc--bản-chất-vấn-đề)
   - 1.1. Bản chất của bài toán Grounding VLM
   - 1.2. Giải phẫu kiến trúc: ViT → Projector → LLM
   - 1.3. Tại sao là thang tọa độ 0–1000? (Scale Invariance)
   - 1.4. Tại sao phải chia tập theo phiên quay (Session)? (Chống Data Leakage)
   - 1.5. Tại sao phải Mask Loss trên Prompt và Ảnh? (Gradient Dilution)
   - 1.6. Tại sao Projector phải giữ float32 và không được lượng tử hóa?
2. [Phần 2: Lộ Trình & Kế Hoạch Thời Gian (Timeline Agenda)](#phần-2-lộ-trình--kế-hoạch-thời-gian-timeline-agenda)
3. [Phần 3: Hướng Dẫn Thực Hiện Từng Bước (Hands-on Execution)](#phần-3-hướng-dẫn-thực-hiện-từng-bước-hands-on-execution)
   - [Bước 0: Thiết lập môi trường & Kiểm tra công cụ](#bước-0-thiết-lập-môi-trường--kiểm-tra-công-cụ)
   - [Bước 1: Tải & Khám phá dữ liệu](#bước-1-tải--khám-phá-dữ-liệu)
   - [Bước 2: Xây dựng hàm chuyển đổi nhãn YOLO sang JSON 0–1000](#bước-2-xây-dựng-hàm-chuyển-đổi-nhãn-yolo-sang-json-01000)
   - [Bước 3: Chạy thí nghiệm E0 (Zero-shot)](#bước-3-chạy-thí-nghiệm-e0-zero-shot)
   - [Bước 4: Pipeline dữ liệu huấn luyện & Collate với Mask Loss](#bước-4-pipeline-dữ-liệu-huấn-luyện--collate-với-mask-loss)
   - [Bước 5: Thiết lập LoRA & Đóng băng/Mở khối (E1, E2, E3)](#bước-5-thiết-lập-lora--đóng-băngmở-khối-e1-e2-e3)
   - [Bước 6: Vòng lặp huấn luyện tối ưu bộ nhớ (Training Loop)](#bước-6-vòng-lặp-huấn-luyện-tối-ưu-bộ-nhớ-training-loop)
   - [Bước 7: Suy luận (Inference) & Chấm điểm chính thức E1, E2, E3](#bước-7-suy-luận-inference--chấm-điểm-chính-thức-e1-e2-e3)
   - [Bước 8: Thí nghiệm Ablation E4 (Cô lập một biến số)](#bước-8-thí-nghiệm-ablation-e4-cô-lập-một-biến-số)
   - [Bước 9: (Bonus +10 điểm) E5 – Huấn luyện tăng cường GRPO với Reward IoU](#bước-9-bonus-10-điểm-e5--huấn-luyện-tăng-cường-grpo-với-reward-iou)
4. [Phần 4: Lời Giải Chi Tiết Cho 10 Câu Hỏi Báo Cáo (REPORT.md)](#phần-4-lời-giải-chi-tiết-cho-10-câu-hỏi-báo-cáo-reportmd)
5. [Phần 5: Bảng Kiểm Định Trước Khi Nộp Bài (Checklist Hoàn Thiện)](#phần-5-bảng-kiểm-định-trước-khi-nộp-bài-checklist-hoàn-thiện)

---

## PHẦN 1: TƯ DUY NGUYÊN LÝ GỐC – BẢN CHẤT VẤN ĐỀ

Trước khi gõ bất kỳ dòng lệnh nào, ta cần trả lời câu hỏi: **Bản chất của những gì ta đang làm là gì?** Khi hiểu rõ bản chất, bạn sẽ không bao giờ mắc phải những lỗi ngớ ngẩn (như quên nạp trọng số, chia nhầm dữ liệu hay tràn bộ nhớ VRAM).

### 1.1. Bản chất của bài toán Grounding VLM
- **Hiện tượng:** Bạn đưa vào 1 bức ảnh chụp mặt đường (kích thước 640×360) và một câu hỏi văn bản: *"Hãy tìm các vị trí hư hại mặt đường"*. Mô hình trả về chuỗi văn bản JSON:
  ```json
  [{"label": "pothole", "bbox_2d": [350, 420, 510, 680]}]
  ```
- **Bản chất nguyên lý:** Mô hình ngôn ngữ lớn (LLM) vốn dĩ **chỉ biết sinh từ ngữ (tokens)** kế tiếp nhau. Mô hình không có một "cánh tay robot" hay "hộp vẽ đồ họa" nào bên trong cả. Việc "định vị vật thể" (Object Grounding) thực chất được quy về bài toán: **Sinh ra các con số tọa độ dưới dạng chuỗi văn bản**.
- **Ví dụ đời thường:** Giống như một người quan sát ngồi trên ô tô gọi điện thoại cho bạn ở nhà. Người đó không thể thò tay qua điện thoại để chỉ ngón tay vào cái ổ gà trên đường, người đó chỉ có thể dùng **lời nói** để mô tả: *"Ở vị trí cách lề trái 35%, cách mép trên 42% có một cái ổ gà rộng 16% và dài 26%"*. VLM làm chính xác việc này thông qua định dạng JSON.

---

### 1.2. Giải phẫu kiến trúc: ViT → Projector → LLM
Một mô hình Vision-Language Model (như Qwen3-VL-4B) không phải là một khối nguyên khối duy nhất, mà là sự hợp tác của 3 thành phần có vai trò sinh học/toán học tách biệt:

```
[Ảnh 640×360]
      ↓
┌──────────────┐
│  Vision      │  (ViT - 415M tham số): "Đôi mắt quan sát"
│  Transformer │  Cắt ảnh thành các mảnh patch (14×14), trích xuất đặc trưng thị giác.
└──────┬───────┘
       │ (Feature maps)
┌──────▼───────┐
│              │  (Projector ≈ 109M tham số): "Người thông dịch viên"
│  Projector   │  - visual.merger (27M): gộp không gian 2×2 patch → 1 token (nén 4 lần).
│              │  - visual.deepstack_merger_list (82M): gom đặc trưng đa tầng (layers 5, 11, 17).
│              │  Chiếu vector thị giác sang không gian vector ngữ nghĩa của LLM.
└──────┬───────┘
       │ (220 visual tokens)
┌──────▼───────┐
│  Large       │  (LLM ≈ 4.0B tham số): "Bộ não suy luận & Thư ký ghi chép"
│  Language    │  Đọc visual tokens cùng prompt văn bản, suy luận quan hệ không gian,
│  Model       │  và tự hồi quy (autoregressive) sinh ra từng ký tự JSON.
└──────────────┘
```

- **Ví dụ đời thường:**
  - **ViT** giống như **Võng mạc mắt**: Tiếp nhận các photon ánh sáng từ mặt đường, cảm nhận các đốm sáng tối, đường nét gồ ghề.
  - **Projector** giống như **Dây thần kinh thị giác & Vùng vỏ não thị giác sơ cấp**: Biến các xung điện quang học từ mắt thành khái niệm trừu tượng mà não hiểu được (ví dụ: chuyển tín hiệu đốm tròn sẫm màu thành ý niệm "vật thể trũng hình tròn").
  - **LLM** giống như **Bán cầu não ngôn ngữ & Bàn tay**: Nhận ý niệm trừu tượng, liên kết với câu hỏi của bạn ("tìm ổ gà"), tổng hợp tư duy và điều khiển bàn tay cầm bút viết ra đúng định dạng sổ sách: `[{"label": "pothole", ...}]`.

---

### 1.3. Tại sao là thang tọa độ 0–1000? (Scale Invariance)
- **Vấn đề của tọa độ Pixel tuyệt đối:** 
  Một bức ảnh dashcam có kích thước 640×360. Nếu bạn dạy mô hình tọa độ tuyệt đối, ổ gà nằm ở `[x=320, y=180]`. Ngày mai, xe khác gắn camera Full HD (1920×1080) chụp lại đúng cái ổ gà đó, tọa độ tuyệt đối nhảy vọt lên `[x=960, y=540]`. 
  Với LLM, số `320` và số `960` là hai token chữ số hoàn toàn khác nhau. Mô hình sẽ bị bối rối vì vị trí vật thể bị phụ thuộc chặt vào độ phân giải camera (không có tính bất biến theo tỉ lệ - scale dependent).
- **Nguyên lý gốc của chuẩn hóa 0–1000:**
  Ta chia chiều rộng và chiều cao ảnh thành 1000 phần bằng nhau (tương đương độ phân giải tỉ lệ 0.1%). Điểm `x=500` luôn luôn là **chính giữa chiều ngang ảnh**, bất kể ảnh rộng 640 pixel, 1920 pixel hay 4K.
  Thang 0–1000 là số nguyên, vừa khớp hoàn hảo với các token chữ số phổ biến của từ điển LLM (các số từ `0` đến `999` thường chỉ chiếm 1 đến 2 token), vừa đảm bảo độ chính xác định vị tới 1/1000 kích thước ảnh (với ảnh 640x360, 1 đơn vị tương đương 0.64 pixel — thừa đủ mịn để đóng khung ổ gà).
- **Ví dụ đời thường:** 
  Khi bạn đọc sách, bạn đánh dấu: *"Tôi đang đọc đến trang 50%"* (chuẩn hóa), thay vì nói *"Tôi đang đọc đến centimet thứ 12 của cuốn sách"* (pixel tuyệt đối). Nếu nói centimet, cuốn sách in khổ to (A4) sẽ cho kết quả trang sách khác hẳn cuốn in bỏ túi (A5), dù cùng một nội dung!

---

### 1.4. Tại sao phải chia tập theo phiên quay (Session)? (Chống Data Leakage)
- **Bản chất dữ liệu Video:**
  Camera hành trình ghi hình liên tục ở tốc độ 25–30 khung hình/giây (fps). Hai khung hình cách nhau 0.1 giây (frame 100 và frame 101) gần như giống nhau 99.9% về góc chiếu, ánh sáng mặt trời, nền đường và chính cái ổ gà đó.
- **Nếu chia ngẫu nhiên (Random Split):**
  Frame 100 rơi vào tập `Train`, frame 101 rơi vào tập `Test`.
  Khi đó, mô hình trong lúc thi (Test) thực chất chỉ đang **nhìn lại bức tranh mà nó vừa được xem đáp án cách đó 0.1 giây trong lúc học (Train)**. Điểm số F1 sẽ đạt 0.95 hay 0.99 cực kỳ đẹp mắt, nhưng đó là **F1 ảo do học vẹt (Memorization / Leakage)**. Khi bạn mang mô hình ra gắn lên một chiếc xe chạy trên đường phố mới, mô hình sẽ hoàn toàn "mù tịt" và sai be bét.
- **Chia theo phiên quay (Session-based Split):**
  Toàn bộ các frame quay trong cùng một khung giờ/ngày của một chuyến đi (cùng 1 `session`) bị khóa chặt và gom chung vào MỘT tập duy nhất. Tập `Test` sẽ là một chuyến đi hoàn toàn khác, ở thời điểm khác, cung đường khác. Mô hình bắt buộc phải học **tính chất đặc trưng của ổ gà** (gờ mép, màu sắc bóng tối, hình dạng nứt) chứ không thể học vẹt bối cảnh xung quanh.
- **Ví dụ đời thường:** 
  Bạn cho học sinh ôn thi môn Toán. Nếu bạn cho học sinh làm đề ôn tập có bài toán: *"Một bể nước dài 2m, rộng 1m..."*, rồi hôm sau đề thi chính thức bạn phát đúng đề đó chỉ đổi mỗi màu mực in: học sinh được 10 điểm nhưng không chứng minh được học sinh biết giải toán. Đề thi thật phải là một bài toán hoàn toàn mới ở một ngữ cảnh khác!

---

### 1.5. Tại sao phải Mask Loss trên Prompt và Ảnh? (Gradient Dilution)
- **Cơ chế Autoregressive Next-Token Prediction:**
  Mô hình ngôn ngữ học bằng cách dự đoán token tiếp theo $P(x_t | x_{<t})$. Hàm mất mát (Loss) là Cross-Entropy tính trên sự khác biệt giữa phân phối dự đoán và token thực tế.
- **Nếu KHÔNG Mask (tính loss trên toàn bộ chuỗi):**
  Độ dài đầu vào bao gồm: ~50 token prompt chỉ dẫn + 220 token ảnh thị giác + ~40 token JSON đáp án.
  Nếu tính loss trên tất cả, hơn 85% gradient sẽ bị lãng phí để ép mô hình học cách: *"Dự đoán lại câu lệnh chỉ dẫn của người dùng"* và *"Tái tạo lại các pixel của bức ảnh"*. Tín hiệu học để định vị vật thể (40 token JSON) bị **pha loãng nghiêm trọng (gradient dilution)**. Mô hình sẽ học rất chậm, dễ sinh ra tật nói nhảm hoặc lặp lại câu hỏi.
- **Giải pháp Mask Loss (gán nhãn `-100`):**
  Trong PyTorch, giá trị nhãn `-100` là quy ước mặc định của `CrossEntropyLoss(ignore_index=-100)`. Khi gặp `-100`, loss tại vị trí đó bằng 0 tuyệt đối và không có gradient nào truyền ngược về. Ta chỉ để nhãn là token ID thật tại phần trả lời của Assistant (từ dấu `[` mở đầu JSON cho đến `<|im_end|>`).
- **Ví dụ đời thường:** 
  Thầy giáo phát đề thi tiếng Anh cho bạn: *"Đề bài: Hãy dịch câu sau sang tiếng Anh..."*. Khi chấm điểm, thầy giáo chỉ chấm các chữ tiếng Anh bạn viết ra. Thầy giáo sẽ không bao giờ trừ điểm hay chấm xem bạn có chép lại câu lệnh *"Đề bài: Hãy dịch câu sau..."* hay không. Ép học sinh chép đề vừa mỏi tay vừa mất tập trung vào việc dịch bài!

---

### 1.6. Tại sao Projector phải giữ float32 và không được lượng tử hóa?
- **Tại sao không lượng tử hóa 4-bit (NF4) ViT & Projector?**
  Lượng tử hóa 4-bit biến các số thực 16-bit thành 16 mức bậc thang rời rạc cố định để tiết kiệm VRAM. Một ma trận đã bị nén 4-bit thì **không thể tính vi phân liên tục để cập nhật trọng số trực tiếp (Backpropagation)** được nữa. Do ở thí nghiệm E2 và E3, ta cần **huấn luyện toàn phần (Full Fine-tuning) Projector** (~109M tham số), nên bắt buộc Projector phải được giữ nguyên vẹn ở dạng số thực (không đưa vào danh sách nén 4-bit, thông qua `llm_int8_skip_modules=["visual", "lm_head"]`).
- **Tại sao Projector phải chuyển sang `float32` khi huấn luyện?**
  Projector là "cổ chai" kết nối giữa ViT và LLM. Nó thực hiện các phép chiếu ma trận tuyến tính và tích tụ gradient từ 36 tầng LLM dội ngược về. Dưới định dạng nửa chính xác `float16`, dải biểu diễn số cực kỳ hẹp (giá trị lớn nhất chỉ khoảng $65504$, giá trị dương nhỏ nhất chỉ $\approx 6 \times 10^{-8}$). Khi gradient tích lũy dồn qua các ma trận nén của merger, nó rất dễ vượt ngưỡng dẫn đến **tràn số (Overflow sinh ra `inf` hoặc `nan`)**. Chuyển toàn bộ tham số projector sang `float32` cung cấp một "vùng đệm an toàn số học" (numerical stability), triệt tiêu hoàn toàn nguy cơ nổ gradient.
- **Ví dụ đời thường:** 
  Projector giống như khớp nối thủy lực chịu lực nén cực mạnh nối giữa đầu kéo xe tải (ViT) và rơ-moóc chở hàng nặng (LLM). Bạn có thể làm vỏ thân xe bằng nhôm nhẹ (float16 để tiết kiệm cân nặng), nhưng riêng chốt nối truyền lực bắt buộc phải đúc bằng thép đặc nguyên khối chịu lực cao (float32), nếu không khớp nối sẽ gãy vụn ngay khi xe tăng tốc!

---

## PHẦN 2: LỘ TRÌNH & KẾ HOẠCH THỜI GIAN (TIMELINE AGENDA)

Tổng thời gian thực hiện ước tính: **3.5 – 4 giờ** trên Google Colab T4 miễn phí.

| Khung thời gian | Giai đoạn | Hành động cụ thể | Sản phẩm đầu ra |
|---|---|---|---|
| **0 – 30 phút** | Chuẩn bị & Dữ liệu | - Cài packages, chạy unit test của đề.<br>- Tải dataset từ Kaggle, chạy `prepare_data.py`.<br>- Khám phá trực quan ảnh và box nhãn. | - `data/images/`, `data/splits/`<br>- 68 tests pass |
| **30 – 60 phút** | Hàm nhãn & E0 | - Viết hàm `yolo_to_box`, `objects_to_answer`.<br>- Nạp model 4-bit, suy luận trên `test.jsonl`.<br>- Chạy `score.py` cho E0. | - `work/e0.jsonl`<br>- `report/evidence/e0.json` |
| **60 – 90 phút** | Pipeline & Mask Loss | - Viết `encode_example` với mask `-100`.<br>- Viết hàm `collate_fn` gom batch.<br>- Viết hàm `apply_setup` cấu hình LoRA và projector. | - Batch tensors chuẩn<br>- Match 252 & 348 modules |
| **90 – 140 phút** | Train & Eval E1 | - Huấn luyện E1 (LoRA LLM, 300 steps ~20 phút).<br>- Suy luận trên `test.jsonl` và chấm điểm. | - `work/e1.jsonl`<br>- `report/evidence/e1.json` |
| **140 – 190 phút** | Train & Eval E2 | - Huấn luyện E2 (+ mở Projector float32).<br>- Lưu adapter + trọng số projector.<br>- Nạp lại đúng cách, suy luận và chấm điểm. | - `work/e2.jsonl`<br>- `report/evidence/e2.json` |
| **190 – 240 phút** | Train & Eval E3 | - Huấn luyện E3 (+ LoRA ViT).<br>- Suy luận và chấm điểm E3. | - `work/e3.jsonl`<br>- `report/evidence/e3.json` |
| **240 – 260 phút** | Ablation E4 | - Chọn 1 biến (khuyên chọn: `rank=4` hoặc `coord_scale=pixel`).<br>- Chạy suy luận/chấm điểm để so sánh. | - `report/evidence/e4_*.json` |
| **260 – 280 phút** | Hoàn thiện & Kiểm tra | - Điền số liệu vào `REPORT.md`.<br>- Trả lời 10 câu hỏi bằng số của mình.<br>- Chạy `scripts/check_submission.py` đạt 100% PASS.<br>- Git commit (≥ 5 commits) và push. | - Repo sạch, sẵn sàng nộp |

---

## PHẦN 3: HƯỚNG DẪN THỰC HIỆN TỪNG BƯỚC (HANDS-ON EXECUTION)

### Bước 0: Thiết lập môi trường & Kiểm tra công cụ

1. **Khởi tạo Colab:** Mở notebook, chọn menu `Runtime` → `Change runtime type` → Chọn **T4 GPU** (15 GB VRAM).
2. **Mount Google Drive** (cực kỳ quan trọng để lưu adapter dự phòng khi Colab bị ngắt phiên):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
```
3. **Cài đặt thư viện đã ghim phiên bản chính xác:**
   ```bash
   pip install -r requirements-colab.txt
   ```
4. **Chạy kiểm thử xác nhận hệ thống hoạt động hoàn hảo:**
   ```bash
   python -m pytest -v
   ```
   *Kết quả mong đợi:* Tất cả 68 bài test tích hợp sẵn đều phải báo `PASSED` (hoặc 1 skipped nếu chưa có raw data).

---

### Bước 1: Tải & Khám phá dữ liệu

1. **Thiết lập Kaggle Credentials qua Colab Secrets:**
   - Bấm vào biểu tượng hình chiếc chìa khóa (Secrets) ở thanh công cụ bên trái Colab.
   - Thêm 2 biến: `KAGGLE_USERNAME` và `KAGGLE_KEY` lấy từ file `kaggle.json` của tài khoản của bạn.
   - Nạp vào môi trường:
     ```python
     import os
     from google.colab import userdata
     os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
     os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
     ```
2. **Tải và phân chia tập dữ liệu:**
   ```bash
   python scripts/prepare_data.py --kaggle
   ```
   *Nguyên lý hoạt động ngầm:* Script sẽ tải tập dữ liệu road damage về `work/raw`, phân tích cấu trúc tên file theo biểu thức chính quy (regex) để gán từng ảnh vào một `session`. Sau đó thuật toán hashing sẽ gán cố định toàn bộ session vào các tập:
   - `train`: 1.422 ảnh (1.074 pothole, 1.750 crack, 691 manhole)
   - `val`: 276 ảnh (66 pothole, 470 crack, 103 manhole)
   - `test`: 311 ảnh (121 pothole, 299 crack, 163 manhole)
   - Sinh file định danh phiên bản `data/splits/meta.json` chứa mã hash cố định `d10-xxxxxxxxxx`.

3. **Code kiểm tra và vẽ thử ảnh lên màn hình:**
   ```python
   import json
   from PIL import Image, ImageDraw
   import matplotlib.pyplot as plt

   with open("data/splits/train.jsonl", "r", encoding="utf-8") as f:
       samples = [json.loads(line) for line in f][:3]

   for item in samples:
       img = Image.open(f"data/{item['image_path']}")
       draw = ImageDraw.Draw(img)
       w, h = item["width"], item["height"]
       for obj in item["objects"]:
           xc, yc, bw, bh = obj["yolo"]
           x1, y1 = (xc - bw/2) * w, (yc - bh/2) * h
           x2, y2 = (xc + bw/2) * w, (yc + bh/2) * h
           draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
       plt.figure(figsize=(6, 4))
       plt.imshow(img)
       plt.title(f"Image ID: {item['image_id']} | Objects: {len(item['objects'])}")
       plt.axis("off")
       plt.show()
   ```

---

### Bước 2: Xây dựng hàm chuyển đổi nhãn YOLO sang JSON 0–1000

Đây là "trái tim" của phần chuẩn bị dữ liệu đầu vào. Ta chuyển tọa độ tâm YOLO `[xc, yc, w, h]` (chuẩn hóa 0..1) sang góc `[x1, y1, x2, y2]` trên thang nguyên `0–1000`, sau đó sắp xếp ổn định.

```python
import json

def yolo_to_box(label: str, yolo: list[float]) -> dict:
    """
    Chuyển đổi 1 nhãn YOLO [xc, yc, w, h] chuẩn hóa 0..1
    sang box Qwen Grounding [x1, y1, x2, y2] trên thang nguyên 0..1000.
    """
    xc, yc, w, h = yolo
    x1 = round((xc - w / 2.0) * 1000.0)
    y1 = round((yc - h / 2.0) * 1000.0)
    x2 = round((xc + w / 2.0) * 1000.0)
    y2 = round((yc + h / 2.0) * 1000.0)
    
    # Kẹp giá trị (clamp) vào đoạn [0, 1000] để chống tràn viền
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}

def objects_to_answer(objects: list[dict]) -> str:
    """
    Chuyển danh sách objects thành chuỗi JSON chuẩn.
    Sắp xếp các box theo thứ tự không gian: từ trên xuống dưới (y1), từ trái sang phải (x1).
    Điều này giúp quá trình học ngôn ngữ của LLM ổn định và hội tụ nhanh hơn rất nhiều.
    """
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)
```

- **Quy chuẩn Prompt chung cho toàn bộ bài lab:**
  ```python
  SYSTEM_PROMPT = (
      "You are a helpful road inspection assistant. Detect all road damage objects in the image. "
      "Output a JSON list containing detected items with 'label' and 'bbox_2d'. "
      "Possible labels: 'pothole', 'crack', 'manhole'. "
      "Coordinates must be integers scaled from 0 to 1000 representing [x1, y1, x2, y2]."
  )
  ```

---

### Bước 3: Chạy thí nghiệm E0 (Zero-shot)

Mục tiêu: Đánh giá khả năng mộc của mô hình trước khi học.

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from pathlib import Path
from tqdm import tqdm

# Cấu hình lượng tử hóa 4-bit NF4
# QUAN TRỌNG: Bỏ qua "visual" và "lm_head" để giữ ở fp16 phục vụ cho E2/E3
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["visual", "lm_head"]
)

model_id = "Qwen/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
model.eval()

# Suy luận trên tập test
import json
with open("data/splits/test.jsonl", "r", encoding="utf-8") as f:
    test_entries = [json.loads(line) for line in f]

Path("work").mkdir(parents=True, exist_ok=True)
out_file = open("work/e0.jsonl", "w", encoding="utf-8")

for item in tqdm(test_entries, desc="Evaluating E0 Zero-shot"):
    img_path = Path("data") / item["image_path"]
    image = Image.open(img_path).convert("RGB")
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": SYSTEM_PROMPT}
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")
    
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    
    response = processor.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
    out_file.write(json.dumps({"image_id": item["image_id"], "raw": response}, ensure_ascii=False) + "\n")

out_file.close()
```

**Chấm điểm E0 ngay lập tức:**
```bash
python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
```
*Kết quả thông thường:* `parse_valid_rate` có thể thấp, F1 gần bằng 0 hoặc rất nhỏ (do mô hình gốc hay trả lời văn bản dài dòng, bịa thêm chữ hoặc dùng pixel tuyệt đối).

---

### Bước 4: Pipeline dữ liệu huấn luyện & Collate với Mask Loss

```python
from torch.utils.data import Dataset

class RoadDamageDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.entries = [json.loads(line) for line in f]
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image = Image.open(Path("data") / item["image_path"]).convert("RGB")
        answer = objects_to_answer(item["objects"])
        
        # 1. Encode prompt của User
        user_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT}
            ]}
        ]
        prompt_enc = self.processor.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        
        # 2. Encode toàn bộ hội thoại (User + Assistant answer)
        full_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]
        full_enc = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = full_enc["input_ids"][0]
        # Tạo nhãn labels: sao chép input_ids, nhưng gán -100 cho toàn bộ phần Prompt
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": full_enc.get("pixel_values"),
            "image_grid_thw": full_enc.get("image_grid_thw")
        }

def collate_fn(batch):
    # Padding input_ids và labels về cùng độ dài lớn nhất trong batch
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = (padded_input_ids != processor.tokenizer.pad_token_id).long()
    
    out = {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels
    }
    
    # Nối các tensor hình ảnh
    if batch[0]["pixel_values"] is not None:
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if batch[0]["image_grid_thw"] is not None:
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        
    return out
```

---

### Bước 5: Thiết lập LoRA & Đóng băng/Mở khối (E1, E2, E3)

Hàm `apply_setup` thực hiện việc bóc tách chính xác các khối cần huấn luyện:

```python
from peft import LoraConfig, get_peft_model
import re

def apply_setup(model, setup_name: str):
    """
    Cấu hình tham số theo setup:
    - E1: LoRA chỉ trên LLM (252 modules)
    - E2: E1 + mở toàn bộ Projector (visual.merger, visual.deepstack_merger_list) ở float32 (~109M params)
    - E3: E2 + LoRA trên cả ViT (348 modules LoRA tổng cộng)
    """
    # 1. Đóng băng toàn bộ model trước
    for p in model.parameters():
        p.requires_grad = False
        
    model.enable_input_require_grads()
    
    # 2. Xác định các target modules cho LoRA
    if setup_name in ("E1", "E2"):
        # Chỉ nhắm vào LLM layers: 36 layers × 7 modules = 252 modules
        target_modules = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    elif setup_name == "E3":
        # Nhắm vào LLM (252) + ViT blocks (24 blocks × 4 modules = 96) = 348 modules
        target_modules = (
            r".*(language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|"
            r"visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2)))"
        )
    else:
        raise ValueError(f"Unknown setup: {setup_name}")
        
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    # 3. Mở Projector nếu là E2 hoặc E3
    if setup_name in ("E2", "E3"):
        for name, param in peft_model.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                # Đưa về float32 để ổn định số học, tránh tràn số khi gradient dội ngược
                param.data = param.data.to(torch.float32)
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"[{setup_name}] Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return peft_model
```

---

### Bước 6: Vòng lặp huấn luyện tối ưu bộ nhớ (Training Loop)

```python
import time
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

def train_setup(peft_model, setup_name: str, max_steps: int = 300):
    train_dataset = RoadDamageDataset("data/splits/train.jsonl", processor)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    optimizer = AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(max_steps * 0.05), num_training_steps=max_steps
    )
    scaler = GradScaler()
    
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.config.use_cache = False
    peft_model.train()
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    step = 0
    grad_accum_steps = 4
    loader_iter = iter(train_loader)
    
    pbar = tqdm(total=max_steps, desc=f"Training {setup_name}")
    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
                
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            with autocast(dtype=torch.float16):
                outputs = peft_model(**batch)
                loss = outputs.loss / grad_accum_steps
                
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is NaN or Inf at step {step}!")
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{accum_loss:.4f}"})
        
    pbar.close()
    train_minutes = (time.time() - start_time) / 60.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    # LƯU TRỌNG SỐ:
    # 1. Lưu adapter LoRA
    save_path = Path(f"work/{setup_name}_adapter")
    peft_model.save_pretrained(save_path)
    
    # 2. CỰC KỲ QUAN TRỌNG: Lưu riêng projector nếu là E2 hoặc E3
    if setup_name in ("E2", "E3"):
        projector_weights = {}
        for name, param in peft_model.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                projector_weights[name] = param.data.cpu()
        torch.save(projector_weights, save_path / "projector_weights.pt")
        print(f"Đã lưu {len(projector_weights)} tensors của Projector vào {save_path}/projector_weights.pt")
        
    return {
        "trainable_params": sum(p.numel() for p in peft_model.parameters() if p.requires_grad),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "train_minutes": round(train_minutes, 2)
    }
```

---

### Bước 7: Suy luận (Inference) & Chấm điểm chính thức E1, E2, E3

```python
def run_eval(setup_name: str, meta_info: dict):
    # Nạp lại model gốc
    raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    # Nạp adapter LoRA
    from peft import PeftModel
    eval_model = PeftModel.from_pretrained(raw_model, f"work/{setup_name}_adapter")
    
    # Nạp lại Projector weights nếu là E2 hoặc E3
    proj_path = Path(f"work/{setup_name}_adapter/projector_weights.pt")
    if proj_path.exists():
        state_dict = torch.load(proj_path, map_location="cuda")
        # Nạp đúng vào model
        eval_model.load_state_dict(state_dict, strict=False)
        print(f"Đã nạp lại thành công projector weights từ {proj_path}")
        
    eval_model.eval()
    
    out_path = f"work/{setup_name.lower()}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(test_entries, desc=f"Inference {setup_name}"):
            img = Image.open(Path("data") / item["image_path"]).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": SYSTEM_PROMPT}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = eval_model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = processor.decode(out[0, prompt_len:], skip_special_tokens=True)
            f.write(json.dumps({"image_id": item["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            
    # Gọi chấm điểm
    import subprocess
    cmd = [
        "python", "scripts/score.py",
        "--pred", out_path,
        "--name", setup_name,
        "--out", f"report/evidence/{setup_name.lower()}.json",
        f"--meta=trainable_params={meta_info['trainable_params']}",
        f"--meta=peak_vram_gb={meta_info['peak_vram_gb']}",
        f"--meta=train_minutes={meta_info['train_minutes']}"
    ]
    subprocess.run(cmd, check=True)
```

---

### Bước 8: Thí nghiệm Ablation E4 (Cô lập một biến số)

Chọn biến **`coord_scale`** (dễ thấy tác động toán học sâu sắc nhất):
1. **Lần chạy 1 (gốc):** Đã có ở E1/E2/E3 với thang `norm1000`.
2. **Lần chạy 2 (thay đổi biến):** Đổi thang toạ độ sang `pixel`.
   - Trong hàm chuyển đổi nhãn: toạ độ là pixel thực tế `[x1, y1, x2, y2]` (không nhân 1000 rồi chia w, h).
   - Prompt sửa thành: *"Coordinates must be integers in pixel coordinates [x1, y1, x2, y2]"*.
   - Huấn luyện mô hình và suy luận ra `work/e4_pixel.jsonl`.
   - Chấm điểm bằng cờ `--coord pixel`:
     ```bash
     python scripts/score.py --pred work/e4_pixel.jsonl --coord pixel --name E4_pixel --out report/evidence/e4_coord_scale_pixel.json --meta trainable_params=... --meta peak_vram_gb=... --meta train_minutes=...
     ```

---

### Bước 9: (Bonus +10 điểm) E5 – Huấn luyện tăng cường GRPO với Reward IoU

1. **Bản chất của GRPO (Group Relative Policy Optimization):**
   Thay vì cần một mạng Critic cồng kềnh (như PPO), GRPO cho mô hình sinh ra một nhóm ($G=4$) câu trả lời khác nhau cho cùng một bức ảnh, sau đó tính reward của từng câu rồi chuẩn hóa điểm (chuẩn hóa tương đối trong nhóm) để cập nhật chính sách.

2. **Hàm tính Reward IoU (Python thuần, có thể test ngay mà không cần GPU):**
   ```python
   from scripts.score import parse_output, iou_xyxy

   def compute_iou_reward(completion: str, gt_boxes: list[dict]) -> float:
       pred_boxes = parse_output(completion)
       if pred_boxes is None:
           return 0.0  # Phạt nặng JSON lỗi cú pháp
           
       if len(gt_boxes) == 0 and len(pred_boxes) == 0:
           return 1.0
           
       # Ghép đôi tham lam theo IoU giảm dần, cùng nhãn lớp
       pairs = []
       for i, g in enumerate(gt_boxes):
           for j, p in enumerate(pred_boxes):
               if g["label"] == p["label"]:
                   iou = iou_xyxy(g["bbox"], p["bbox"])
                   if iou >= 0.5:
                       pairs.append((iou, i, j))
                       
       pairs.sort(reverse=True, key=lambda x: x[0])
       used_gt, used_pred = set(), set()
       total_iou = 0.0
       
       for iou, i, j in pairs:
           if i not in used_gt and j not in used_pred:
               used_gt.add(i)
               used_pred.add(j)
               total_iou += iou
               
       # Công thức: 0.2 điểm cho cú pháp hợp lệ + 0.8 điểm cho độ chuẩn xác IoU
       # Mẫu số max(n_gt, n_pred, 1) trừng phạt thích đáng việc model "bịa" ra quá nhiều box rác (False Positives)
       denominator = max(len(gt_boxes), len(pred_boxes), 1)
       reward = 0.2 + 0.8 * (total_iou / denominator)
       return round(reward, 4)
   ```

---

## PHẦN 4: LỜI GIẢI CHI TIẾT CHO 10 CÂU HỎI BÁO CÁO (REPORT.MD)

Dưới đây là đáp án chi tiết từ góc nhìn nguyên lý gốc. **Lưu ý:** Khi điền vào `REPORT.md`, bạn thay thế các số liệu trong ngoặc nhọn bằng số liệu thực tế đo được từ các file `report/evidence/*.json` của bạn.

---

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân rã tham số nguyên lý gốc:**
  - **ViT (Vision Transformer):** $\approx 415\text{M}$ tham số (gồm 24 khối transformer xử lý ảnh).
  - **Projector:** $\approx 109\text{M}$ tham số, chia thành 2 cấu phần cốt lõi:
    - `visual.merger`: $\approx 27\text{M}$ tham số (nén 4 patch $2 \times 2$ thành 1 visual token).
    - `visual.deepstack_merger_list`: $\approx 82\text{M}$ tham số (lấy đặc trưng từ các tầng trung gian 5, 11, 17 của ViT để giữ lại chi tiết vật thể nhỏ).
  - **LLM (Language Model):** $\approx 4.0\text{B}$ tham số (gồm 36 tầng giải mã ngôn ngữ kèm embedding và lm_head).
  - **Tổng tham số toàn mô hình:** $\approx 4.44\text{B}$ tham số.
- **Khối cập nhật trong từng thí nghiệm:**
  - **E1:** Chỉ cập nhật LoRA trên LLM (252 modules gồm $q, k, v, o, gate, up, down\_proj$ trên 36 tầng). Số tham số huấn luyện $\approx <số\_E1>$. Toàn bộ ViT và Projector bị đóng băng hoàn toàn.
  - **E2:** Cập nhật LoRA trên LLM (giống E1) **CỘNG VỚI** mở toàn bộ Projector ở float32 ($\approx 109\text{M}$ tham số). Số tham số huấn luyện tăng thêm đúng $\approx 109\text{M}$ so với E1 ($\approx <số\_E2>$).
  - **E3:** Giữ nguyên setup của E2 **CỘNG VỚI** gắn thêm LoRA vào 24 khối của ViT (thêm 96 modules LoRA gồm $qkv, proj, linear\_fc1, linear\_fc2$). Tổng số modules gắn LoRA là $252 + 96 = 348$. Số tham số huấn luyện $\approx <số\_E3>$.

---

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Số liệu:** `parse_valid_rate` đạt $<điền\_ví\_dụ:\ 0.150\ hoặc\ 0.420>$.
- **Hai lỗi phổ biến nhất:**
  1. **Lỗi format và sinh văn bản thừa (Chatty / Markdown fence):** Mô hình gốc được huấn luyện để trò chuyện với con người, nên nó thường chèn thêm lời giải thích xã giao (*"Here is the detection result:"*) hoặc quên đóng thẻ markdown ```` ```json ```` khiến parser JSON chuẩn bị lỗi.
  2. **Bịa nhãn sai và dùng sai thang tọa độ (Scale hallucination):** Mô hình gốc chưa được fine-tune trên tập dữ liệu chuyên biệt nên thường tự tiện trả về tọa độ pixel thực tế (như x=550, y=340 trên ảnh 640x360) hoặc đưa ra nhãn ngoài danh mục (như "road", "car", "asphalt") thay vì chỉ trả về 3 nhãn quy định.
- **Ví dụ output thô:**
  ```text
  Certainly! Here are the detected objects on the road:
  ```json
  [{"label": "street_crack", "bbox_2d": [320, 180, 450, 220]}]
  ```
  Hope this helps you inspect the road!
  ```

---

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả:** F1 sẽ tụt thẳng về **0.000** (hoặc cực kỳ sát 0), dù mô hình vẫn định vị trúng phóc vật thể!
- **Nguyên lý gốc:**
  - Ảnh có kích thước $640 \times 360$. Một cái ổ gà nằm ở giữa ảnh có toạ độ pixel là $[x_1=256, y_1=144, x_2=384, y_2=216]$.
  - Trên thang chuẩn hoá $0–1000$, toạ độ đúng là $[x_1=400, y_1=400, x_2=600, y_2=600]$.
  - Nếu mô hình xuất ra theo pixel mà bộ chấm điểm đọc theo thang $1000$, nó sẽ hiểu rằng box chỉ nằm ở góc trên cùng bên trái của khung hình ($25.6\%$ chiều rộng và $14.4\%$ chiều cao).
  - Diện tích giao nhau (Intersection) giữa 2 box này bằng 0 (hoặc cực nhỏ), dẫn đến $\text{IoU} < 0.5$ và toàn bộ dự đoán bị coi là False Positive (dự đoán sai) và False Negative (bỏ sót).
- **Kiểm chứng:** Đã được chứng minh trực tiếp trong unit test `test_pixel_vs_norm1000_scale_trap` của file `tests/test_score.py`: cùng một nhãn dự đoán, khi chấm bằng cờ `--coord pixel` thì đạt $\text{F1} = 1.0$, nhưng chấm bằng `--coord norm1000` thì $\text{F1} = 0.0$ tuyệt đối dù cú pháp JSON hợp lệ 100%.

---

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Nguyên lý gốc:**
  - Mô hình học thông qua cơ chế tự hồi quy sinh token tiếp theo. Mục tiêu thực tế là mô hình phải sinh ra **chuỗi JSON trả lời** khi nhìn thấy **ảnh và câu hỏi**.
  - Phần prompt và 220 tokens ảnh là **dữ liệu ngữ cảnh đầu vào (Conditioning context)** được cung cấp sẵn cho mô hình, chứ không phải mục tiêu cần dự đoán.
- **Hậu quả nếu tính loss trên cả prompt và ảnh:**
  - **Pha loãng Gradient (Gradient Dilution):** Prompt và ảnh chiếm tới ~270 tokens, trong khi câu trả lời JSON chỉ chiếm khoảng 30–50 tokens. Hơn 85% năng lượng tính toán và gradient sẽ bị phân tán vào việc ép LLM học cách "tái tạo câu lệnh tiếng Anh" và "tái tạo các vector điểm ảnh".
  - **Học sai mục đích (Task Distortion):** Mô hình sẽ dễ bị hiện tượng "nhại lại đề bài" (parroting), lặp lại prompt của người dùng thay vì sinh ra cấu trúc toạ độ. Tốc độ hội tụ trên bài toán định vị sẽ bị chậm đi rõ rệt.

---

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Nguyên lý gốc về lượng tử hóa:**
  - Lượng tử hóa NF4 (4-bit) nén các tham số thành 16 mức rời rạc cố định để tiết kiệm VRAM. Các tham số 4-bit này bị đóng băng vì không thể tính đạo hàm vi phân để cập nhật trọng số liên tục.
  - Nếu lượng tử hoá ViT, toàn bộ các tầng của Projector (nằm bên trong `visual`) cũng sẽ bị nén 4-bit, khiến việc mở Full Fine-tuning ở E2 và E3 là bất khả thi về mặt toán học.
  - **Cách xử lý:** Thêm tham số `llm_int8_skip_modules=["visual", "lm_head"]` vào `BitsAndBytesConfig`. Tham số này ra lệnh cho bitsandbytes giữ nguyên ViT và Projector ở định dạng số thực `float16` chuẩn.
- **Tại sao Projector phải ở float32:**
  - Projector là "cổ chai" dồn nén thông tin từ ViT để bơm vào LLM, đồng thời nhận toàn bộ gradient dội ngược từ 36 tầng LLM.
  - Ở định dạng nửa chính xác `float16`, dải số biểu diễn cực kỳ hạn hẹp ($[-65504, 65504]$). Quá trình cộng dồn gradient qua các phép nhân ma trận dày đặc của `merger` rất dễ dẫn đến tràn số (Overflow), sinh ra các giá trị `NaN` hoặc `Inf` làm sập quá trình huấn luyện.
  - Chuyển tham số Projector sang `float32` tạo ra vùng an toàn số học giúp việc lan truyền ngược và cập nhật tối ưu hoàn toàn ổn định.

---

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Nguyên lý rò rỉ dữ liệu (Data Leakage):**
  - Dữ liệu dashcam thực chất là các video được cắt thành các frame liên tiếp ở tốc độ cao (25–30 fps). Hai frame cách nhau 0.1 giây gần như trùng khít về góc nhìn, màu sắc mặt đường, ánh sáng thời tiết và chính cái ổ gà đó.
  - Nếu chia ngẫu nhiên, frame $t$ nằm ở Train và frame $t+0.1$ nằm ở Test. Mô hình chỉ việc "học vẹt" bối cảnh khung hình của frame $t$ là có thể dễ dàng đoán trúng frame $t+0.1$ mà không cần học được bản chất hư hại mặt đường. Điểm F1 trên tập test sẽ cao vút một cách giả tạo, nhưng mô hình sẽ thất bại hoàn toàn khi đem ra đường thật.
- **Bảng phân bố số lượng ảnh và box từng lớp (lấy từ `data/splits/meta.json`):**
  - **Train (1.422 ảnh):** 1.074 pothole, 1.750 crack, 691 manhole (Tổng: 3.515 boxes).
  - **Val (276 ảnh):** 66 pothole, 470 crack, 103 manhole (Tổng: 639 boxes).
  - **Test (311 ảnh):** 121 pothole, 299 crack, 163 manhole (Tổng: 583 boxes).
  *(Nhận xét: Do chia theo phiên quay nguyên vẹn, phân bố lớp ở tập Val và Test bị lệch tự nhiên so với Train, phản ánh đúng thực tế thu thập dữ liệu ngoài đời thực).*

---

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh số liệu thực nghiệm:**
  - *E1 (LoRA LLM):* F1 $\approx <F1\_E1>$, VRAM đỉnh $\approx <VRAM\_E1>\text{ GB}$, Thời gian $\approx <Min\_E1>\text{ phút}$.
  - *E2 (+ Projector):* F1 $\approx <F1\_E2>$, VRAM đỉnh $\approx <VRAM\_E2>\text{ GB}$, Thời gian $\approx <Min\_E2>\text{ phút}$.
  - *E3 (+ LoRA ViT):* F1 $\approx <F1\_E3>$, VRAM đỉnh $\approx <VRAM\_E3>\text{ GB}$, Thời gian $\approx <Min\_E3>\text{ phút}$.
- **Phân tích bản chất:**
  - **E1 → E2 (Mở Projector):** Đem lại bước nhảy vọt lớn nhất và "đáng đồng tiền bát gạo" nhất. Do vật thể hư hại mặt đường (đặc biệt là vết nứt) có diện tích trung vị cực nhỏ (chỉ $\approx 0.7\%$ diện tích ảnh), việc mở Projector giúp bộ thông dịch học cách chuyển đổi các chi tiết cục bộ từ các tầng nông (thông qua DeepStack) sang LLM tốt hơn hẳn. Mức tăng VRAM là vừa phải (chủ yếu do tham số float32 của Projector).
  - **E2 → E3 (Thêm LoRA ViT):** Cải thiện thêm một phần F1 (nhất là ở lớp crack và pothole nhỏ), nhưng chi phí VRAM và thời gian tăng lên đáng kể do phải tính toán lan truyền ngược qua toàn bộ 24 khối ViT khổng lồ. Nếu khoảng tin cậy CI95 của E3 chồng lấn (overlap) nhiều với E2, sự cải thiện này có thể chưa mang ý nghĩa thống kê vượt trội so với chi phí tài nguyên bỏ ra.

---

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy Bootstrap 95% (CI95):**
  - Tập test chỉ có 311 ảnh (với 583 box). Khi chạy Bootstrap 1000 lần, khoảng tin cậy CI95 thường có biên độ rộng khoảng $\pm 0.03$ đến $\pm 0.05$ (ví dụ: $[0.482, 0.564]$, tức độ rộng dải tin cậy lên tới $\approx 0.08$).
- **Giới hạn kết luận khoa học:**
  - Nếu mô hình A có F1 = 0.52 và mô hình B có F1 = 0.54, nhưng khoảng tin cậy của A là $[0.48, 0.56]$ và B là $[0.50, 0.58]$, hai khoảng này **chồng lấn nhau rất lớn**.
  - Ta **không thể khẳng định chắc chắn 100%** rằng B vượt trội hơn A ngoài thực tế. Sự chênh lệch 0.02 đó hoàn toàn có thể chỉ là do may mắn ngẫu nhiên trong việc phân mẫu của tập test 311 ảnh. Kết luận chỉ thực sự vững chắc khi sự chênh lệch vượt ra ngoài biên độ của khoảng tin cậy (như bước nhảy vọt giữa E1 so với E0).

---

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có F1 kém nhất:** Lớp `crack` (vết nứt) hoặc `manhole` (tùy thuộc vào số liệu thực tế của bạn, thông thường là `crack`).
- **Nguyên nhân nguyên lý gốc:**
  1. **Hình dạng hình học và kích thước cực nhỏ:** Vết nứt có đặc thù là mảnh, dài ngoằn ngoèo và diện tích thực chỉ chiếm $\approx 0.7\%$ khung hình. ViT gom các cụm patch 14×14 lại nên rất dễ làm mờ các chi tiết mảnh này. Đóng khung chữ nhật (Bounding Box) quanh một vết nứt ngoằn ngoèo thường chứa nhiều nền đường hơn là vết nứt, khiến IoU tính toán dễ bị phạt dưới ngưỡng 0.5.
  2. **Độ tương phản thấp:** Vết nứt màu xám đen nằm trên nền nhựa đường xám dưới ánh sáng chói chang của camera hành trình có độ tương phản thị giác rất yếu, dễ bị mô hình nhầm với bóng râm của cây cối hoặc vệt lốp xe.
  3. **Lớp Manhole (Nắp cống):** Tuy có hình dạng tròn/vuông rõ ràng nhưng số lượng mẫu trong tập dữ liệu lại ít nhất, dẫn đến mô hình ít được tiếp xúc với đa dạng góc nhìn nắp cống.

---

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến số lựa chọn:** `coord_scale` (so sánh giữa `norm1000` và `pixel`) *[hoặc rank=4 vs rank=64]*:
  - Giá trị A: `norm1000` → F1 = $<F1\_norm1000>$, CI95 = $<CI\_A>$.
  - Giá trị B: `pixel` → F1 = $<F1\_pixel>$, CI95 = $<CI\_B>$.
- **Phân tích kết quả:**
  - Mô hình dùng thang `norm1000` học ổn định và hội tụ tốt hơn hẳn do tính bất biến tỉ lệ không gian và tương thích với từ điển token của LLM.
- **Độ tin cậy của kết luận:**
  - Kết luận này là **hoàn toàn đáng tin cậy** vì khoảng cách chênh lệch giữa hai setup vượt xa biên độ khoảng tin cậy CI95, phản ánh đúng sự khác biệt về mặt bản chất biểu diễn dữ liệu của mô hình ngôn ngữ.

---

## PHẦN 5: BẢNG KIỂM ĐỊNH TRƯỚC KHI NỘP BÀI (CHECKLIST HOÀN THIỆN)

Trước khi đóng gói nộp bài, hãy chạy danh sách kiểm tra nghiêm ngặt này:

- [ ] **Chạy kiểm tra tự động:**
  ```bash
  python scripts/check_submission.py
  ```
  *Bắt buộc phải hiển thị 0 FAIL và 0 WARN (hoặc chỉ WARN git nếu chưa tạo repo).*
- [ ] **Bằng chứng đầy đủ:** Thư mục `report/evidence/` có đủ:
  - `e0.json`, `e1.json`, `e2.json`, `e3.json`
  - Ít nhất một file `e4_*.json`
- [ ] **Khớp phiên bản dữ liệu:** Tất cả các file bằng chứng đều có cùng `dataset_version` khớp với `data/splits/meta.json`.
- [ ] **Không còn chuỗi giữ chỗ:** File `REPORT.md` không còn bất kỳ ký tự `<điền` nào; các bảng số liệu đã điền đủ số thực tế.
- [ ] **Ảnh trực quan:** Có ít nhất 6 ảnh trực quan hóa kết quả dự đoán đúng/sai lưu trong repo/báo cáo.
- [ ] **Lịch sử Git:** Repo có ít nhất **5 commit** thể hiện quá trình làm bài từng bước.
- [ ] **Repo sạch sẽ:** Đã kiểm tra `.gitignore` để chắc chắn **KHÔNG COMMIT**:
  - Thư mục dữ liệu: `data/`
  - Thư mục nháp: `work/`
  - Khóa bí mật: `.env`
  - File trọng số nặng: `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`

---
*Bản hướng dẫn được xây dựng theo chuẩn First Principles Thinking – Chúc bạn hoàn thành xuất sắc bài lab Day 10!*
