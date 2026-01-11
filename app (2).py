import google.generativeai as genai
genai.configure(api_key="AIzaSyAO2yhWsZA5fk3H9-N4BMoH3c9oz4PQSnE")

import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini API key (thay bằng key thật của bạn từ Google AI Studio)
genai.configure(api_key="AIzaSyBzrosbf3CCJlm9AIuGbApVit4yUPcWLQQ")

# ================== CẤU HÌNH ==================
JSON_FILE = "/content/drive/RAG/all_procedures_normalized.json"  # Đường dẫn file JSON (sau chunk rule-based)
CHROMA_DB_PATH = "chroma_db"  # Thư mục lưu vector DB
COLLECTION_NAME = "dichvucong_rag"
GEMINI_MODEL = "gemini-2.5-flash"  # Hoặc "gemini-1.5-pro"

@st.cache_resource
def get_embedding_function():
    EMBEDDING_MODEL = "BAAI/bge-m3"  # Model embedding tiếng Việt
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return embedding_function

@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_func = get_embedding_function()

    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func  # cần để query đúng
        )
        #st.success(f"Collection '{COLLECTION_NAME}' đã load từ {CHROMA_DB_PATH}")
    except Exception as e:
        st.error(f"Không tìm thấy collection '{COLLECTION_NAME}' trong {CHROMA_DB_PATH}: {e}")
        collection = None

    return collection
# --- Load collection 1 lần ---
collection = load_collection()

def query_rag(query: str, chat_history: list, top_k: int):
    # Retrieval với top_k động
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['hierarchy']}]\\n{doc}\\n(Nguồn: {meta['url']})")

    context = "\\n\\n".join(context_parts)

    prompt = f"""
Bạn là trợ lý tư vấn thủ tục hành chính công của Việt Nam.
Bạn chỉ trả lời câu hỏi.
KHÔNG được viết lại, diễn đạt lại hay sửa đổi câu hỏi của người dùng.
KHÔNG nhắc lại câu hỏi.
PHẠM VI ÁP DỤNG:
- Ưu tiên tư vấn các thủ tục hành chính liên quan đến trẻ em dưới 6 tuổi.
- Nếu CONTEXT không đề cập rõ độ tuổi nhưng nội dung thuộc thủ tục thường áp dụng cho trẻ em,
  bạn được phép trả lời dựa trên thông tin hiện có và nêu rõ phạm vi áp dụng nếu được đề cập.

NGUYÊN TẮC TRẢ LỜI:
- Chỉ sử dụng thông tin có trong CONTEXT bên dưới.
- Không sử dụng kiến thức bên ngoài.
- Không tự bổ sung thông tin không có trong CONTEXT.
- Không tự thay đổi câu hỏi của người dùng.

CÁCH TRẢ LỜI:
- Chỉ trả lời các nội dung LIÊN QUAN TRỰC TIẾP đến câu hỏi.
- Có thể tổng hợp nhiều đoạn trong CONTEXT nếu chúng cùng mô tả một thủ tục.
- Trình bày ngắn gọn, rõ ràng, đúng trọng tâm.

TRƯỜNG HỢP KHÔNG TRẢ LỜI:
Chỉ trả lời đúng câu sau nếu:
- CONTEXT hoàn toàn không chứa thông tin liên quan đến câu hỏi.

Câu trả lời trong trường hợp này PHẢI CHÍNH XÁC:
"Không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

YÊU CẦU ĐỊNH DẠNG:
- Trả lời bằng tiếng Việt.
- Nếu có nhiều ý, trình bày bằng gạch đầu dòng hoặc đánh số.
- Giữ nguyên trích dẫn nguồn nếu có trong CONTEXT.

    Context:
    {context}

    Câu hỏi: {query}

    Trả lời bằng tiếng Việt, có đánh số nếu là danh sách, và trích dẫn nguồn rõ ràng (tên block, URL):
    """

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt, stream=True)

    return response

# ================== GIAO DIỆN CHÍNH ==================
st.set_page_config(
    page_title="Chatbot tư vấn thủ tục hành chính trẻ em dưới 6 tuổi",
    page_icon="🤖",
    layout="centered"
)

# ================== TIÊU ĐỀ ==================
st.title("🤖 Chatbot tư vấn thủ tục hành chính trẻ em dưới 6 tuổi")
st.markdown(
    "Hỗ trợ tư vấn **đăng ký khai sinh – đăng ký thường trú – cấp thẻ BHYT** "
    "cho **trẻ em dưới 6 tuổi** dựa trên dữ liệu chính thống."
)

# ================== SIDEBAR ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 📜 Lịch sử trò chuyện")

    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"**👤 Người dùng:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Chatbot:** {msg['content'][:150]}...")
            st.divider()
    else:
        st.caption("Chưa có cuộc trò chuyện nào.")

    if collection:
        try:
           data = collection.get(include=["metadatas"])
           metadatas = data.get("metadatas", [])

           source_files = set()

           for meta in metadatas:
              if not meta:
                continue

              file_name = meta.get("source_file", "").strip()
              if file_name:
                source_files.add(file_name)

        except Exception as e:
            st.error(f"Lỗi khi tải file dữ liệu: {e}")
    else:
        st.caption("Chưa tải được dữ liệu vector.")

    if collection:
        metadatas = collection.get(
            include=["metadatas"],
            limit=10
        )["metadatas"]

        seen = set()
        for meta in metadatas:
            url = meta.get("url", "")
            code = meta.get("procedure_code", "")
            if url and url not in seen:
                st.markdown(f"- **{code}**: [Link]({url})")
                seen.add(url)
    else:
        st.caption("Chưa tải được dữ liệu.")

    st.divider()

    st.markdown("## ℹ️ Thông tin hệ thống")
    st.write(f"📦 Vector DB: {COLLECTION_NAME}")
    st.write(f"🧩 Số chunk: {collection.count() if collection else 0}")
    st.write(f"🤖 LLM: {GEMINI_MODEL}")
    st.write("📐 Embedding: BAAI/bge-m3")
    st.caption("Dữ liệu được load từ file JSON.")

# ================== KHỞI TẠO LỊCH SỬ CHAT ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================== HIỂN THỊ LỊCH SỬ CHAT ==================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ================== INPUT TỪ USER ==================
prompt = st.chat_input(
    "Hỏi về thủ tục hành chính cho trẻ em dưới 6 tuổi "
    "(ví dụ: Giấy khai sinh có cấp bản điện tử không?)"
)

if prompt:
    # Lưu câu hỏi
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ================== GỌI BACKEND (GIỮ NGUYÊN) ==================
    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty()

        try:
            response = query_rag(prompt, st.session_state.messages, top_k)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response)
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Lỗi khi gọi Gemini: {str(e)}"
            message_placeholder.error(full_response)


    # Lưu câu trả lời
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
