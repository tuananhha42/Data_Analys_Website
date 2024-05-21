import streamlit as st
st.set_page_config(page_title="Dashboard", page_icon='📊', layout='wide')

from predict import *
from descriptive import *

# CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
    
def main():
    
    try:
        with open("../note.txt", "r", encoding="utf-8") as file:
            note_content = file.read().split("*")
    except FileNotFoundError:
        note_content = "Không tìm thấy tệp văn bản."

    # Expander hướng dẫn sử dụng
    with st.expander("📢HƯỚNG DẪN SỬ DỤNG PHẦN MỀM"):
        st.write(note_content)
        
    # st.session_state.data = upload_data()
    # if st.session_state.data is not None:
        
        
    #     sideBar(st.session_state.data)
        
        
    with st.sidebar:
        st.sidebar.image("../img/image.png", caption='Data Analys')
        selected = option_menu(
            menu_title="Main Menu",
            options=['🏠Home', "📊Phân tích mô tả", "📈Phân tích dự báo", '⏱Lịch sử dự báo'],
            icons=["🏠", "📊", "📈", '⏱'],
            menu_icon="cast",
            default_index=0
        )

    try:
        if selected == "🏠Home":
            st.session_state.data = upload_data()
        if selected == "📊Phân tích mô tả" and 'data' in st.session_state:
            st.subheader("🔍Bắt đầu quá trình phân tích mô tả")
            show_statistics(st.session_state.data)
            visualize_data(st.session_state.data)
        if selected == "📈Phân tích dự báo" and 'data' in st.session_state:
            st.subheader("💡Bắt đầu quá trình phân tích dự báo")
            preprocessing(st.session_state.data)
        if selected == "⏱Lịch sử dự báo":
            history()
    except:
        pass

    if 'data' not in st.session_state:
        st.subheader("Chưa có dữ liệu để phân tích!")

if __name__ == "__main__":
    main()                 
