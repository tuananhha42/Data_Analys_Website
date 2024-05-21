import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import seaborn as sns
import matplotlib.pyplot as plt
from predict import *
import time
import toml
import plotly.graph_objects as go
import plotly.figure_factory as ff
from streamlit_extras.metric_cards import style_metric_cards
import random




def upload_data():
    st.subheader('⬆️Vui lòng tải lên dữ liệu')

    with st.expander("⬆️Tải lên dữ liệu"):
        uploaded_file = st.file_uploader("Chọn tệp dữ liệu CSV hoặc Excel", type=['csv', 'xlsx', 'txt'])
        if uploaded_file is not None:
            try:
                # Hiển thị thanh tiến trình
                progress_bar = st.progress(0)
                
                # Giả lập việc tải lên trong 1 giây
                for percent_complete in range(100):
                    time.sleep(0.000)  # Giảm thời gian sleep để tăng tốc độ hiển thị thanh tiến trình
                    progress_bar.progress(percent_complete + 1,text='Đang tải lên dữ liệu')
                
                if uploaded_file.name.endswith('csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('xlsx'):
                    st.session_state.df = pd.read_excel(uploaded_file)
                else:
                    st.session_state.df = pd.read_csv(uploaded_file, delimiter='\t')
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi đọc tệp: {e}")
    if "df" in st.session_state:
        st.subheader('📄Dữ liệu đã tải lên')
        st.write(st.session_state.df)
        return st.session_state.df



def show_statistics(df):
    st.session_state.numerical_columns = df.select_dtypes(include=['number']).columns
    ### deccor màn chính
    total1,total2,total3,total4,total5,total6 = st.columns(6, gap='small')
    with total1:
        st.session_state.select_column = st.selectbox('Chọn 1 cột',label_visibility='collapsed', options=st.session_state.numerical_columns, key='select_colums')
        if not st.session_state.select_column:
            st.session_state.select_column = "None"
            st.warning("Vui lòng chọn một cột.")
        st.metric(label= "Cột được chọn", value=f"{st.session_state.select_column}")

    if st.session_state.select_column != "None":
        st.session_state.min = (df[st.session_state.select_column]).min()
        st.session_state.max = (df[st.session_state.select_column]).max()
        st.session_state.sum = (df[st.session_state.select_column]).sum()
        st.session_state.mean = (df[st.session_state.select_column]).mean()
        st.session_state.count_null = (df[st.session_state.select_column]).isnull().sum()
        with total2:
            st.info(f"Min of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Min of {st.session_state.select_column}", value=f"{st.session_state.min:,.0f}")
        with total3:
            st.info(f"Max of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Max of {st.session_state.select_column}", value=f"{st.session_state.max:,.0f}")
        with total4:
            st.info(f"Sum of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Sum of {st.session_state.select_column}", value=f"{st.session_state.sum:,.0f}")
        with total5:
            st.info(f"Mean of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Mean of {st.session_state.select_column}", value=f"{st.session_state.mean:,.0f}")
        with total6:
            st.info(f"Count null of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Count null of {st.session_state.select_column}", value=f"{st.session_state.count_null:,.0f}")
    else:
        with total2:
            st.info(f"Min of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Min of {st.session_state.select_column}", value=f"{0}")
        with total3:
            st.info(f"Max of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Max of {st.session_state.select_column}", value=f"{0}")
        with total4:
            st.info(f"Sum of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Sum of {st.session_state.select_column}", value=f"{0}")
        with total5:
            st.info(f"Mean of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Mean of {st.session_state.select_column}", value=f"{0}")
        with total6:
            st.info(f"Count null of {st.session_state.select_column}",icon='📌')
            st.metric(label= f"Count null of {st.session_state.select_column}", value=f"{0}")
    style_metric_cards(border_color="blue",background_color='#00172B',border_left_color='blue')
    st.markdown("""---""")


    with st.sidebar:
        st.title('Phân tích mô tả')
    # Kiểm tra xem DataFrame có dữ liệu dạng số không
    
    if not st.session_state.numerical_columns.empty:
        # Tạo expander mới trong sidebar
        with st.sidebar.expander("Analys Data"):
            st.markdown("*Lưu ý: Phân tích dựa trên những thuộc tính dạng số*")
            # Thêm radio button cho các thống kê muốn hiển thị
            selected_statistic = st.radio("Chọn thống kê", options=['Min', 'Max', 'Median', 'Mean','Mode', 'Count Null', 'Q1,Q2,Q3,IQR', 'Variance', 'Standard Deviation', 'Thống kê describe'])  
            selected_columns = st.multiselect("Chọn cột", options=st.session_state.numerical_columns, key='select_analys')
            if not selected_columns:
                st.warning("Vui lòng chọn ít nhất một cột.")
                return
            
            # Xử lý hiển thị dữ liệu theo radio button được chọn
            if df is not None:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("Analys"):
                        if selected_statistic == 'Thống kê describe':
                            st.session_state.statistics_result = df[selected_columns].describe(include='all')
                        elif selected_statistic == 'Min':
                            st.session_state.statistics_result = df[selected_columns].min()
                        elif selected_statistic == 'Max':
                            st.session_state.statistics_result = df[selected_columns].max()
                        elif selected_statistic == 'Mean':
                            st.session_state.statistics_result = df[selected_columns].mean()
                        elif selected_statistic == 'Median':
                            st.session_state.statistics_result = df[selected_columns].median()
                        elif selected_statistic == 'Mode':
                            st.session_state.statistics_result = df[selected_columns].mode().iloc[0]
                        elif selected_statistic == 'Count Null':
                            st.session_state.statistics_result = df[selected_columns].isnull().sum()
                        elif selected_statistic == 'Q1,Q2,Q3,IQR':
                            quantiles = df[selected_columns].quantile([0.25, 0.5, 0.75])
                            quantiles.loc['IQR'] = quantiles.loc[0.75] - quantiles.loc[0.25]
                            st.session_state.statistics_result = quantiles
                        elif selected_statistic == 'Variance':
                            st.session_state.statistics_result = df[selected_columns].var()
                        elif selected_statistic == 'Standard Deviation':
                            st.session_state.statistics_result = df[selected_columns].std()
                        else:
                            st.session_state.statistics_result = getattr(df[selected_columns], selected_statistic.lower())()

                with col2:
                    if st.button("Reset Analys"):
                        st.session_state.statistics_result = None
                    
        # Hiển thị kết quả phân tích trên giao diện chính
        if "statistics_result" in st.session_state:
            st.write('Thống kê dữ liệu')
            st.write(st.session_state.statistics_result)
    else:
        st.warning("Không có cột nào chứa dữ liệu dạng số trong dữ liệu.")       

def visualize_data(df):
    st.session_state.numerical_columns = df.select_dtypes(include=['number']).columns
    st.session_state.df_number = df[st.session_state.numerical_columns]
    if not st.session_state.numerical_columns.empty:
        with st.sidebar.expander("Trực quan hóa dữ liệu"):
            st.markdown("*Lưu ý: Trực quan hóa dựa trên những thuộc tính dạng số*")
            selected_columns = st.multiselect("Chọn cột", options=st.session_state.numerical_columns, key='select_visualize_data')
            st.session_state.selected_columns = selected_columns
            # Kiểm tra xem selected_columns đã được chọn hay chưa
            if not selected_columns:
                # Nếu chưa chọn, gán selected_columns là một cột ngẫu nhiên từ numerical_columns
                selected_columns = [random.choice(st.session_state.numerical_columns)]
            
            st.session_state.selected_columns = selected_columns

        row1_col1, row1_col2 = st.columns(2)
        # if st.sidebar.button("Xem biểu đồ"):
            
        if "selected_columns" in st.session_state:
            with row1_col1:
                st.subheader("📈Biểu đồ đường")
                fig_line = go.Figure()
                # Tạo biểu đồ đường cho từng cột
                for column in st.session_state.df_number[st.session_state.selected_columns]:
                    fig_line.add_trace(go.Scatter(x=st.session_state.df_number.index, y=st.session_state.df_number[column], mode='lines', name=column))
                fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_line)

            with row1_col2:
                st.subheader("📊Biểu đồ cột")
                fig_bar = go.Figure()
                # Tạo biểu đồ cột cho từng cột
                for column in st.session_state.df_number[st.session_state.selected_columns]:
                    fig_bar.add_trace(go.Bar(x=st.session_state.df_number.index, y=st.session_state.df_number[column], name=column))
                fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar)

            st.markdown("""---""")

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.subheader("📦Biểu đồ box")
                fig_box = go.Figure()
                # Tạo biểu đồ box cho từng cột
                for column in st.session_state.df_number[st.session_state.selected_columns]:
                    fig_box.add_trace(go.Box(y=st.session_state.df_number[column], name=column))
                fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_box)

            with row2_col2:
                st.subheader("🚥Biểu đồ scatter")
                fig_scatter = go.Figure()
                # Tạo biểu đồ scatter cho từng cột
                for column in st.session_state.df_number[st.session_state.selected_columns]:
                    fig_scatter.add_trace(go.Scatter(x=st.session_state.df_number.index, y=st.session_state.df_number[column], mode='markers', name=column))
                fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter)

            st.markdown("""---""")

            row3_col1, row3_col2 = st.columns(2)
            with row3_col1:
                st.subheader("🌡Biểu đồ heatmap")
                fig_heatmap = go.Figure(data=go.Heatmap(z=st.session_state.df_number.corr().values,
                                                        x=selected_columns,
                                                        y=selected_columns,
                                                        colorscale='Viridis'))

                # Tinh chỉnh layout của biểu đồ
                fig_heatmap.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickmode='array', tickvals=list(range(len(selected_columns))), ticktext=selected_columns),
                    yaxis=dict(tickmode='array', tickvals=list(range(len(selected_columns))), ticktext=selected_columns),
                )

                st.plotly_chart(fig_heatmap)

            with row3_col2:
                st.subheader("🍩Biểu đồ tròn (Pie chart)")
                # Tính tỷ lệ phần trăm của từng cột
                column_percentages = (st.session_state.df_number[st.session_state.selected_columns].sum() / st.session_state.df_number[st.session_state.selected_columns].sum().sum()) * 100
                fig_pie = px.pie(values=column_percentages, names=column_percentages.index)
                st.plotly_chart(fig_pie)
    else:
        st.warning("Không có cột nào chứa dữ liệu dạng số trong dữ liệu.")

    
    #---------------------------------------------Phần trên đã done-------------------------------------------   
# def sideBar(data):
#     with st.sidebar:
#         st.sidebar.image("../img/image.png",caption='Data Analys')
#         selected = option_menu(
#             menu_title = "Main Menu",
#             options=["📊Phân tích mô tả", "📈Phân tích dự báo", '⏱Lịch sử dự báo'],
#             icons=["📊","📈",'⏱'],
#             menu_icon= "cast",
#             default_index=0
#         )
#     if selected == "📊Phân tích mô tả":
#         st.subheader("🔍Bắt đầu quá trình phân tích mô tả")
#         show_statistics(data)
#         visualize_data(data)
#     if selected == "📈Phân tích dự báo":
#         st.subheader("💡Bắt đầu quá trình phân tích dự báo")
#         preprocessing(data)
#     if selected == "⏱Lịch sử dự báo":
#         history()
   

# def main():
   
#     try:
#         with open("../note.txt", "r", encoding="utf-8") as file:
#             note_content = file.read().split("*")
#     except FileNotFoundError:
#         note_content = "Không tìm thấy tệp văn bản."

#     # Expander hướng dẫn sử dụng
#     with st.expander("📢HƯỚNG DẪN SỬ DỤNG PHẦN MỀM"):
#         st.write(note_content)
        
#     st.session_state.data = upload_data()
#     if st.session_state.data is not None:
        
        
#         sideBar(st.session_state.data)
        
        
#         # preprocessing(st.session_state.data)

# if __name__ == "__main__":
#     main()                 


