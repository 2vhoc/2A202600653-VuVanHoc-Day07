# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Văn Học  
**Nhóm:** C2  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding gần cùng hướng, tức là hai đoạn text có nội dung hoặc ý nghĩa gần nhau trong không gian biểu diễn. Với retrieval, score cao thường cho thấy chunk có khả năng liên quan đến query.

**Ví dụ HIGH similarity:**
- Sentence A: Rivastigmine is linked to Alzheimer disease.
- Sentence B: Donepezil is linked to Alzheimer disease.
- Tại sao tương đồng: Cả hai câu đều nói về thuốc và Alzheimer disease.

**Ví dụ LOW similarity:**
- Sentence A: Morphine sulfate is a narcotic analgesic.
- Sentence B: Python is used for backend services.
- Tại sao khác: Một câu thuộc domain y tế, câu còn lại thuộc lập trình.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối, nên phù hợp hơn khi so sánh ý nghĩa giữa văn bản. Với embedding text, hai câu cùng ý nghĩa có thể có độ lớn vector khác nhau nhưng vẫn gần hướng nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Công thức: `ceil((doc_length - overlap) / (chunk_size - overlap))`  
Tính: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`  
Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Tính: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, nên số chunk tăng từ 23 lên 25. Overlap nhiều hơn giúp giữ ngữ cảnh giữa hai chunk liền kề, nhưng làm tăng số chunk cần embed và search.

---

## 2. Document Selection — Nhóm (10 điểm)

**Domain:** Y tế (thuốc) - medical formulary retrieval và drug-disease knowledge base.

Nhóm C2 chọn domain này vì đã có tìm hiểu về dữ liệu thuốc trong buổi lab trước và có sẵn nguồn data phù hợp để thử nghiệm. Bộ tài liệu có nhiều tên thuốc, tier, payer, năm, và quan hệ thuốc-bệnh nên rất phù hợp để kiểm thử retrieval có metadata filtering. Đây cũng là domain dễ thấy ảnh hưởng của chunking: nếu cắt sai ranh giới dòng thuốc, kết quả có thể thiếu tier, quantity limit, hoặc bệnh liên quan.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | caremark-oct2013.txt | https://github.com/ericminikel/cnsdrugs.git | 2013 | doc_type=formulary_guide; payer=Caremark; year=2013; language=en |
| 2 | healthalliance-2013.txt | https://github.com/ericminikel/cnsdrugs.git | 2235 | doc_type=formulary_guide; payer=Health Alliance; year=2013; language=en |
| 3 | humana_2014_wi.txt | https://github.com/ericminikel/cnsdrugs.git | 59004 | doc_type=formulary_guide; payer=Humana; year=2014; language=en |
| 4 | uhc-cns-drugs-pdf-list.txt | https://github.com/ericminikel/cnsdrugs.git | 17552 | doc_type=formulary_guide; payer=UnitedHealthcare; year=2013; language=en |
| 5 | medical_knowledge_base.txt | Local generated medical knowledge base | 319651 | doc_type=drug_disease_knowledge_base; payer=N/A; year=N/A; language=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source_file | string | humana_2014_wi.txt | Giúp trace kết quả về đúng tài liệu gốc. |
| doc_type | string enum | formulary_guide, drug_disease_knowledge_base | Giúp phân loại cấu trúc văn bản và lọc giữa danh mục thuốc với knowledge base quan hệ thuốc-bệnh. |
| payer | string enum | Humana, Caremark, UnitedHealthcare | Giúp query theo hãng bảo hiểm/formulary, tránh lấy nhầm payer khác. |
| year | integer/string | 2013, 2014 | Hỗ trợ lọc theo thời gian hiệu lực vì chính sách bảo hiểm y tế thay đổi theo từng năm. |
| language | string enum | en, vi | Tránh trộn tài liệu tiếng Anh và tiếng Việt khi query có ngôn ngữ cụ thể. |
| domain | string enum | medical_insurance, cns_pharmacology | Thu hẹp phạm vi tìm kiếm theo không gian kiến thức, tăng precision và giảm nhiễu từ tài liệu không liên quan. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `python3 run.py`, trong đó baseline dùng `ChunkingStrategyComparator().compare(text)` trên toàn bộ 5 tài liệu nhóm.

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| caremark-oct2013.txt | FixedSizeChunker (`fixed_size`) | 14 | 190.2 | Trung bình: ổn định kích thước nhưng dễ cắt ngang dòng thuốc |
| caremark-oct2013.txt | SentenceChunker (`by_sentences`) | 1 | 2013.0 | Thấp với formulary: file ít dấu câu nên chunk quá dài |
| caremark-oct2013.txt | RecursiveChunker (`recursive`) | 11 | 175.2 | Tốt: ưu tiên đoạn/dòng, hợp với danh sách thuốc |
| healthalliance-2013.txt | FixedSizeChunker (`fixed_size`) | 15 | 195.7 | Trung bình: ổn định kích thước nhưng dễ cắt ngang dòng thuốc |
| healthalliance-2013.txt | SentenceChunker (`by_sentences`) | 1 | 2235.0 | Thấp với formulary: file ít dấu câu nên chunk quá dài |
| healthalliance-2013.txt | RecursiveChunker (`recursive`) | 12 | 178.0 | Tốt: ưu tiên đoạn/dòng, hợp với danh sách thuốc |
| humana_2014_wi.txt | FixedSizeChunker (`fixed_size`) | 394 | 199.6 | Trung bình: có thể chia giữa cụm tên thuốc và rule QL/PA |
| humana_2014_wi.txt | SentenceChunker (`by_sentences`) | 49 | 1202.6 | Trung bình/thấp: giữ nhiều dòng liên quan nhưng chunk khá lớn |
| humana_2014_wi.txt | RecursiveChunker (`recursive`) | 331 | 173.5 | Tốt: giữ dòng thuốc ngắn và dễ inspect |
| uhc-cns-drugs-pdf-list.txt | FixedSizeChunker (`fixed_size`) | 117 | 199.6 | Trung bình: kích thước đều nhưng có thể cắt ngang section |
| uhc-cns-drugs-pdf-list.txt | SentenceChunker (`by_sentences`) | 9 | 1948.7 | Thấp: chunk rất lớn vì tài liệu là danh sách nhiều dòng |
| uhc-cns-drugs-pdf-list.txt | RecursiveChunker (`recursive`) | 94 | 178.6 | Tốt: phù hợp cấu trúc section/dòng |
| medical_knowledge_base.txt | FixedSizeChunker (`fixed_size`) | 2131 | 200.0 | Trung bình: có thể cắt ngang một statement thuốc-bệnh |
| medical_knowledge_base.txt | SentenceChunker (`by_sentences`) | 926 | 342.7 | Tốt với KB: giữ nguyên các câu mô tả quan hệ thuốc-bệnh |
| medical_knowledge_base.txt | RecursiveChunker (`recursive`) | 2777 | 113.1 | Tốt nhưng hơi nhỏ, đôi khi tách quá vụn |

### Strategy Của Tôi

**Loại:** RecursiveChunker (`chunk_size=500`)

**Mô tả cách hoạt động:**  
RecursiveChunker chia văn bản theo nhiều cấp độ ưu tiên thay vì cắt trực tiếp theo số ký tự. Thuật toán cố gắng tách theo đoạn văn (`\n\n`) trước, sau đó đến dòng (`\n`), câu (`. `), khoảng trắng, và cuối cùng mới fallback sang cắt theo kích thước cố định. Cách này giúp giữ lại cấu trúc tự nhiên của tài liệu và giảm tình trạng cắt ngang ngữ nghĩa.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Domain của nhóm chứa các tài liệu có độ dài và định dạng rất khác nhau, từ formulary dạng danh sách dòng đến knowledge base dài bằng tiếng Việt. RecursiveChunker tận dụng cấu trúc sẵn có trong tài liệu để bảo toàn ngữ cảnh nhưng vẫn tạo ra các chunk đủ nhỏ cho embedding và retrieval. Kết quả benchmark cho thấy strategy này hoạt động ổn định trên cả tài liệu ngắn lẫn tài liệu lớn.

**Code snippet (nếu custom):**
```python
strategy = RecursiveChunker(chunk_size=500)
```

### So Sánh: Strategy của tôi vs Baseline

Performance được đo bằng `python3 run.py`: mỗi strategy chunk toàn bộ 5 tài liệu, embed từng chunk bằng hashing lexical embedder, lưu vào `EmbeddingStore`, chạy cùng 5 benchmark queries, rồi chấm top-3 relevance.

| Strategy | Stored Chunks | Build Time (ms) | Avg Query Time (ms) | Max Query Time (ms) | Precision@3 | Retrieval Score (/10) |
|-----------|--------------:|----------------:|--------------------:|--------------------:|------------:|----------------------:|
| FixedSizeChunker(500, overlap=50) | 892 | 186.05 | 5.92 | 22.24 | 1.00 | 10 |
| SentenceChunker(3 sentences) | 986 | 185.29 | 6.49 | 28.80 | 1.00 | 10 |
| **RecursiveChunker(500) của tôi** | 868 | 168.84 | 5.66 | 21.61 | 1.00 | **10** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Trần Tiến Đạt | RecursiveChunker (`recursive`) | 8.4 | Giữ ngữ cảnh tốt, chunk ổn định, phù hợp cho retrieval | Avg length thấp hơn mục tiêu nên số chunk tăng nhẹ |
| Vũ Văn Học (tôi) | RecursiveChunker (`recursive`) | 8.5 | Chunk count thấp, avg length gần kích thước mục tiêu, tận dụng cấu trúc tài liệu tốt | Có thể lệch khi chuyển sang domain khác có cấu trúc quá khác |
| Hồ Trọng Nhật Minh | RecursiveChunker (`recursive`) | 8.3 | Giữ ngữ cảnh tốt, chunk ổn định, phù hợp cho retrieval | Cải thiện chưa nhiều so với FixedSize trên một số query |
| Nguyễn Đức Thành | RecursiveChunker (`recursive`) | 8.5 | Đơn giản khi giải thích và dễ kiểm soát kích thước chunk | Nếu cấu hình chưa tốt vẫn có thể cắt mất ngữ nghĩa |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
RecursiveChunker là strategy phù hợp nhất cho domain này. Các tài liệu trong bộ dữ liệu y tế có cấu trúc rất đa dạng, từ danh mục thuốc, tài liệu bảo hiểm đến knowledge base dài; RecursiveChunker chia văn bản theo nhiều cấp độ hơn FixedSizeChunker, đồng thời tránh tạo ra các chunk quá lớn như SentenceChunker trên các file ít dấu câu.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk` — approach:**  
Tôi dùng regex `(?<=[.!?])\s+` để tách câu theo dấu kết thúc câu rồi strip whitespace. Sau đó gom từng nhóm `max_sentences_per_chunk` câu thành một chunk, xử lý edge case text rỗng bằng cách trả về list rỗng.

**`RecursiveChunker.chunk` / `_split` — approach:**  
RecursiveChunker có base case là text rỗng hoặc text ngắn hơn `chunk_size`. Nếu text quá dài, thuật toán thử separator theo thứ tự `\n\n`, `\n`, `. `, space, rồi fallback cắt fixed-size nếu không còn separator. Cách này ưu tiên giữ paragraph/line trước khi phải cắt nhỏ hơn.

### EmbeddingStore

**`add_documents` + `search` — approach:**  
`add_documents` tạo record gồm id, content, metadata, embedding rồi lưu vào in-memory list. `search` embed query, tính dot product với embedding của từng record, thêm `score`, rồi sort giảm dần để lấy top-k.

**`search_with_filter` + `delete_document` — approach:**  
`search_with_filter` lọc metadata trước, sau đó mới search trên tập record đã lọc để giảm nhiễu. `delete_document` xóa tất cả record có `metadata["doc_id"]` trùng với doc_id cần xóa và trả về boolean cho biết có xóa được gì không.

### KnowledgeBaseAgent

**`answer` — approach:**  
Agent retrieve top-k chunks từ store, render từng chunk thành context có source, rồi build prompt yêu cầu LLM trả lời dựa trên context. Sau đó agent gọi `llm_fn(prompt)` và trả về kết quả.

### Test Results

```text
pytest tests/ -v
42 passed in 0.07s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Các score được tính bằng `compute_similarity(_mock_embed(sentence_a), _mock_embed(sentence_b))`.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Rivastigmine is linked to Alzheimer disease. | Donepezil is linked to Alzheimer disease. | high | 0.1356 | Đúng một phần |
| 2 | Buspirone is listed as Buspar for anxiety. | Buspirone appears in an anxiety formulary. | high | 0.0672 | Đúng một phần |
| 3 | ABILIFY 10 MG TABLET has a quantity limit. | Humana lists ABILIFY tablets with QL rules. | high | -0.1391 | Sai |
| 4 | Morphine sulfate is a narcotic analgesic. | Python is used for backend services. | low | -0.1207 | Đúng |
| 5 | Metadata filters narrow retrieval by payer. | Zolpidem is a sleep aid drug. | low | -0.1024 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Pair 3 bất ngờ nhất vì hai câu cùng nói về ABILIFY và quantity limit nhưng mock embedding cho score âm. Điều này cho thấy `_mock_embed` chỉ hữu ích để test code deterministic, không phản ánh semantic similarity thật; khi đánh giá retrieval thực tế nên dùng embedder semantic hoặc ít nhất lexical embedder phù hợp domain.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries bằng `python3 run.py`. Benchmark dùng toàn bộ 5 tài liệu medical mới tải lên, `RecursiveChunker(500)`, metadata filtering, và hashing lexical embedder local để không phụ thuộc API key.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What relationship does Rivastigmine have with Alzheimer's disease in the medical knowledge base? | Rivastigmine has a DM clinical relationship with Alzheimer's disease and DrugBank ID DB00989. |
| 2 | In Health Alliance 2013, which anxiety medication is listed as Buspar? | Buspirone is listed as buspirone (Buspar) under Anxiety. |
| 3 | What tier and quantity limit is ABILIFY 10 MG TABLET in Humana 2014 Wisconsin? | ABILIFY 10 MG TABLET is listed as MO tier 4 with QL 30 per 30 days. |
| 4 | UnitedHealthcare Morphine Sulfate Solution Oral Roxanol MS Contin | The UHC CNS list includes Morphine Sulfate oral solution, Roxanol, MS Contin, and related Morphine Sulfate entries. |
| 5 | In Caremark October 2013, what note is listed for buspirone? | Buspirone is listed with the note NP = 7.5 mg. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What relationship does Rivastigmine have with Alzheimer's disease in the medical knowledge base? | medical_knowledge_base.txt: Rivastigmine, DB00989, DM, Alzheimer's disease | 0.114 | Yes | Rivastigmine có quan hệ DM với Alzheimer's disease, DrugBank DB00989. |
| 2 | In Health Alliance 2013, which anxiety medication is listed as Buspar? | healthalliance-2013.txt: top-3 chứa chunk Anxiety có buspirone (Buspar) | 0.121 | Yes | Buspirone được liệt kê là buspirone (Buspar) trong mục Anxiety. |
| 3 | What tier and quantity limit is ABILIFY 10 MG TABLET in Humana 2014 Wisconsin? | humana_2014_wi.txt: top-3 chứa ABILIFY 10 MG TABLET MO 4 QL (30 per 30 days) | 0.353 | Yes | ABILIFY 10 MG TABLET thuộc MO 4, QL 30 per 30 days. |
| 4 | UnitedHealthcare Morphine Sulfate Solution Oral Roxanol MS Contin | uhc-cns-drugs-pdf-list.txt: Narcotic Analgesics, Morphine Sulfate, Roxanol, MS Contin | 0.453 | Yes | UHC list chứa Morphine Sulfate oral solution, Roxanol và MS Contin trong narcotics. |
| 5 | In Caremark October 2013, what note is listed for buspirone? | caremark-oct2013.txt: top-3 chứa buspirone, NP = 7.5 mg | 0.323 | Yes | Buspirone có note NP = 7.5 mg. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Qua phần trình bày của các thành viên, tôi nhận ra việc lựa chọn chunking strategy không chỉ ảnh hưởng đến số lượng chunk mà còn ảnh hưởng trực tiếp đến chất lượng retrieval. Dù cùng sử dụng RecursiveChunker, mỗi người có cách đánh giá và phân tích trade-off giữa chunk size, overlap và khả năng giữ ngữ cảnh khác nhau. Điều này giúp tôi hiểu rõ hơn cách tối ưu chunking cho từng loại dữ liệu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
Một số nhóm đã thử áp dụng metadata filtering và xây dựng các bộ benchmark query đa dạng hơn thay vì chỉ kiểm tra retrieval đơn thuần. Điều này cho thấy chất lượng của hệ thống RAG không chỉ phụ thuộc vào embedding hay chunking mà còn phụ thuộc vào cách tổ chức dữ liệu và thiết kế phương pháp đánh giá. Tôi học được tầm quan trọng của việc xây dựng benchmark phù hợp với domain thực tế.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Nếu làm lại, tôi sẽ bổ sung thêm nhiều nguồn dữ liệu y tế có cấu trúc khác nhau để tăng độ đa dạng của knowledge base. Tôi cũng sẽ chuẩn hóa metadata chi tiết hơn như loại thuốc, nhóm bệnh, nguồn dữ liệu, payer, tier và quantity_limit để tận dụng metadata filtering hiệu quả hơn trong quá trình retrieval.

**Failure case / giới hạn quan sát được:**  
RecursiveChunker đạt top-3 relevant cho cả 5 query trong benchmark, nhưng một số query có top-1 chưa phải dòng chứa đáp án trực tiếp; đáp án đúng nằm trong top-3 sau khi metadata filter. Nguyên nhân là tài liệu formulary có nhiều dòng thuốc ngắn, lặp lại các token như `MO`, `QL`, `Tier`, khiến lexical embedding đôi khi ưu tiên chunk có từ chung. Cải thiện tốt nhất là custom chunker theo từng dòng thuốc hoặc thêm metadata như `drug_name`, `tier`, `quantity_limit`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
