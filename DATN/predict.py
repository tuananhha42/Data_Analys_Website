import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
import numpy as np
import time
from datetime import datetime
import json

st.markdown(
    """
    <style>
    .green-text {
        font-size: 20px;
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Hàm xử lý giá trị khuyết
def handle_missing_values(df, method):
    if method == "Mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "Median":
        df.fillna(df.median(), inplace=True)
    elif method == "Mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    return df

# Hàm xử lý giá trị ngoại lai
def handle_outliers(df, column, min_value, max_value):
    df[column] = df[column].apply(lambda x: min_value if x < min_value else (max_value if x > max_value else x))
    return df

# Function to scale the data based on the selected method
def scale_data(X_train, X_test, scaling_method):
    if scaling_method == "MinMax":
        scaler = MinMaxScaler()
    elif scaling_method == "Standard":
        scaler = StandardScaler()
    elif scaling_method == "Norm":
        scaler = Normalizer()
    else:
        return X_train, X_test, None
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reset index and reassign original index
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, scaler

# Hàm để mã hóa dữ liệu dạng số thành dạng nhị phân (0 và 1) bằng phương pháp One-Hot Encoding
def encode_categorical_data(df, columns):
    encoded_df = df
    for col in columns:
        encoded_df = pd.get_dummies(encoded_df, columns=[col], drop_first=False)
        # Sử dụng drop_first=True để tránh multicollinearity
    return encoded_df

# Hàm để chuyển DataFrame dạng One-Hot về dạng nhị phân (0 và 1)
def convert_to_binary(df):
    binary_df = df
    binary_df = binary_df.replace(False, 0)
    binary_df[binary_df != 0] = 1
    return binary_df

def train_regression_model(selected_model, X_train, X_test, Y_train, Y_test):
    model = None
    if selected_model == "Linear Regression":
        model = LinearRegression()
    elif selected_model == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
    elif selected_model == "Lasso":
        model = Lasso()
    elif selected_model == 'Ridge':
        model = Ridge()

    if model:
        # Huấn luyện mô hình
        model.fit(X_train, Y_train)

        # Đánh giá mô hình
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        # Trả về các thông số đánh giá
        return model,Y_pred, {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2}
    else:
        st.warning("Vui lòng chọn một mô hình hồi quy để huấn luyện và đánh giá.")

def train_classification_model(selected_model, X_train, X_test, Y_train, Y_test):
    model = None
    if selected_model == "Logistic Regression":
        model = LogisticRegression()
    elif selected_model == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    elif selected_model == "Naive Bayes":
        model = GaussianNB()
    elif selected_model == "Support Vector Machine":
        model = SVC()
    if model:
        # Huấn luyện mô hình
        model.fit(X_train, Y_train)

        # Đánh giá mô hình
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        classification_rep = classification_report(Y_test, Y_pred)

        # Trả về các thông số đánh giá
        return model,Y_pred, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Classification Report": classification_rep}
    else:
        st.warning("Vui lòng chọn một mô hình phân loại để huấn luyện và đánh giá.")


def preprocessing(df):
    if "df" not in st.session_state:
        # Upload dữ liệu và hiển thị DataFrame
        st.session_state.df = df  # Thay "data.csv" bằng tên file của bạn
    else:
        st.session_state.df = df
    st.write("DataFrame ban đầu:")
    st.write(st.session_state.df)

    st.session_state.df_number_collect = st.session_state.df.select_dtypes(include=['number'])
    st.session_state.categorical = st.session_state.df.select_dtypes(include=['object'])
    st.sidebar.title('🖍Xử lý dữ liệu')

    # # Expander cho xử lý giá trị khuyết
    # with st.sidebar.expander("Xử lý giá trị khuyết"):
    #     st.markdown("*Lưu ý: Thực hiện các bước tiền xử lý dữ liệu*")
    #     missing_handling = st.radio("Phương pháp", ["None", "Mean", "Median", "Mode"], index=0)
    
    # if missing_handling != "None":
    #     st.session_state.missing_handling = missing_handling
    # # Xử lý khi nút "Xử lý khuyết" được nhấn
    # if "missing_handling" in st.session_state:
    #     if st.session_state.missing_handling != "None":
    #         st.session_state.df_number = handle_missing_values(st.session_state.df_number, st.session_state.missing_handling)
    #         st.write("DataFrame sau khi xử lý khuyết:")
    #         st.write(st.session_state.df_number)

    total1,total2,total3 = st.columns(3)
    # Expander cho xử lý giá trị khuyết
    with st.sidebar.expander("Xử lý giá trị khuyết"):
        st.markdown("*Lưu ý: Thực hiện các bước tiền xử lý dữ liệu*")
        st.session_state.missing_handling = st.selectbox("Phương pháp", ["None", "Mean", "Median", "Mode"], index=0)
        if st.session_state.missing_handling != "None":
            if st.button("Xử lý khuyết"):
                st.session_state.df_number_pre = handle_missing_values(st.session_state.df_number_collect, st.session_state.missing_handling)
            # if st.button("Hoàn tác"):
            #     st.session_state.df_number_pre = st.session_state.df_number_collect
    with total1:
        if "df_number_pre" in st.session_state:
            st.write("🌙DataFrame sau khi xử lý khuyết:")
            st.write(st.session_state.df_number_pre)

    # Expander cho xử lý giá trị ngoại lai (tùy chọn)
    with st.sidebar.expander("Xử lý giá trị ngoại lai (tùy chọn)"):
        st.markdown("*Lưu ý: Chọn cột và nhập giá trị lớn nhất và nhỏ nhất*")
        numerical_columns = st.session_state.df_number_collect.select_dtypes(include=['number']).columns
        outlier_options = ["None"] + list(numerical_columns)
        selected_column_outlier = st.selectbox("Chọn cột để xử lý ngoại lai", options=outlier_options, index=0)
        if selected_column_outlier != "None":
            min_outlier = st.number_input("Giá trị nhỏ nhất:", value=st.session_state.df_number_collect[selected_column_outlier].min(), step=0.01)
            max_outlier = st.number_input("Giá trị lớn nhất:", value=st.session_state.df_number_collect[selected_column_outlier].max(), step=0.01)

            if isinstance(min_outlier, float) and isinstance(max_outlier, float):

                if st.button("Thực hiện Xử lý"):
                    if "df_number_pre" in st.session_state:
                        st.session_state.df_outliers_removed_temp = handle_outliers(st.session_state.df_number_pre, selected_column_outlier, min_outlier, max_outlier)
                    else:
                        st.session_state.df_outliers_removed_temp = handle_outliers(st.session_state.df_number_collect, selected_column_outlier, min_outlier, max_outlier)
                    # st.write("DataFrame sau khi xử lý ngoại lai:")
                    # st.write(st.session_state.df_outliers_removed_temp)

                # if st.button("Hoàn tác"):
                #     if "df_number_pre" in st.session_state:
                #         st.session_state.df_outliers_removed_temp = st.session_state.df_number_pre
                #     else:
                #         st.session_state.df_outliers_removed_temp = st.session_state.df_number_collect
    with total2:
        # Hiển thị DataFrame sau khi xử lý
        if "df_outliers_removed_temp" in st.session_state:
            st.write("📤DataFrame sau khi xử lý ngoại lai:")
            st.write(st.session_state.df_outliers_removed_temp)
        # Multiselect cho việc mã hóa dữ liệu dạng số

    with st.sidebar.expander("Mã hóa dữ liệu dạng chữ"):
        st.markdown("*Lưu ý: Chọn cột để mã hóa*")
        numerical_columns = st.session_state.categorical.columns
        columns_to_encode = st.multiselect("Chọn cột để mã hóa", options=numerical_columns)
    
        if columns_to_encode and st.button("Mã hóa"):
            st.session_state.categorical_encoded = encode_categorical_data(st.session_state.categorical, columns_to_encode)
            # st.write("DataFrame sau khi mã hóa:")
            # st.write(st.session_state.categorical_encoded)
        elif not columns_to_encode:
            st.warning("Vui lòng chọn ít nhất một cột để mã hóa.")
    
        # if st.button("Hoàn tác"):
        #     st.session_state.categorical_encoded = st.session_state.categorical

    with total3:
        # Hiển thị DataFrame sau khi mã hóa trên màn hình chính
        if "categorical_encoded" in st.session_state:
            st.write("🔁DataFrame sau khi mã hóa:")
            st.session_state.categorical_encoded = convert_to_binary(st.session_state.categorical_encoded)
            st.write(st.session_state.categorical_encoded)
    

    # Nút để nối hai bảng dữ liệu
    if st.sidebar.button("Nối 2 loại dữ liệu"):
        if "categorical_encoded" in st.session_state:
            if "df_outliers_removed_temp" in st.session_state:
                st.session_state.joined_df_3_fun = pd.concat([st.session_state.df_outliers_removed_temp, st.session_state.categorical_encoded], axis=1)
            elif "df_outliers_removed_temp" not in st.session_state and "df_number_pre" in st.session_state:
                st.session_state.joined_df_2_fun = pd.concat([st.session_state.df_number_pre, st.session_state.categorical_encoded], axis=1)
            elif "df_outliers_removed_temp" not in st.session_state and "df_number_pre" not in st.session_state:
                st.session_state.joined_df_2_fun = pd.concat([st.session_state.df_number_collect, st.session_state.categorical_encoded], axis=1)

        elif "categorical_encoded" not in st.session_state:
            if "df_outliers_removed_temp" in st.session_state:
                st.session_state.joined_df_3_fun = pd.concat([st.session_state.df_outliers_removed_temp, st.session_state.categorical], axis=1)
            elif "df_outliers_removed_temp" not in st.session_state and "df_number_pre" in st.session_state:
                st.session_state.joined_df_2_fun = pd.concat([st.session_state.df_number_pre, st.session_state.categorical], axis=1)
            elif "df_outliers_removed_temp" not in st.session_state and "df_number_pre" not in st.session_state:
                st.session_state.joined_df_2_fun = pd.concat([st.session_state.df_number_collect, st.session_state.categorical], axis=1)
     
    if "joined_df_3_fun" in st.session_state:
        st.write("DataFrame sau nối:")
        st.write(st.session_state.joined_df_3_fun)
    elif "joined_df_2_fun" in st.session_state:
        st.write("DataFrame sau nối:")
        st.write(st.session_state.joined_df_2_fun)
    

    #Expander cho việc chọn tập X và Y
    with st.sidebar.expander("Chọn tập X và Y"):
        if "df_outliers_removed_temp" in st.session_state and "categorical_encoded" in st.session_state and "joined_df_3_fun" in st.session_state:
            # Cả df_outliers_removed_temp và categorical_encoded đều đã được tạo
            st.session_state.joined_df = st.session_state.joined_df_3_fun
        elif "df_outliers_removed_temp" in st.session_state and "categorical_encoded" not in st.session_state and 'categorical' in st.session_state:
            # Chỉ df_outliers_removed_temp được tạo
            st.session_state.joined_df = pd.concat([st.session_state.df_outliers_removed_temp,st.session_state.categorical], axis=1)
        elif "categorical_encoded" in st.session_state and "df_outliers_removed_temp" not in st.session_state and "joined_df_2_fun" in st.session_state:
            # Chỉ categorical_encoded được tạo
            st.session_state.joined_df = st.session_state.joined_df_2_fun
        elif "df_number_pre" in st.session_state and  "df_outliers_removed_temp" not in st.session_state and "categorical_encoded" not in st.session_state and 'categorical' in st.session_state and "joined_df_2_fun" in st.session_state:
            # Bằng dữ liệu gốc nếu ko có dữ liệu mới nào được tạo
            st.session_state.joined_df = st.session_state.joined_df_2_fun
        elif "df_number_pre" not in st.session_state and "df" in st.session_state:
            st.session_state.joined_df = st.session_state.df
        elif "df_number_pre" in st.session_state and "categorical" not in st.session_state:
            st.session_state.joined_df = st.session_state.df_number_pre

        # st.write(st.session_state.joined_df)
        # Multiselect cho việc chọn biến độc lập (X)
        st.markdown("*Lưu ý: Chọn các biến độc lập (X)*")
        selected_independent_variables = st.multiselect("Chọn biến độc lập (X)", options=list(st.session_state.joined_df.columns))
        
        # Loại bỏ các biến đã chọn trong X khỏi danh sách các biến để chọn Y
        dependent_variable_options = [col for col in st.session_state.joined_df.columns if col not in selected_independent_variables]
        
        # Selectbox cho việc chọn biến phụ thuộc (Y)
        st.markdown("*Lưu ý: Chọn biến phụ thuộc (Y)*")
        selected_dependent_variable = st.selectbox("Chọn biến phụ thuộc (Y)", options=["None"] + dependent_variable_options, index=0)

        # Kiểm tra xem người dùng đã chọn biến phụ thuộc (Y) chưa
        is_variable_selected = False
        if selected_dependent_variable != "None":
            is_variable_selected = True

        # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
        if is_variable_selected:
            if st.button("Chia tập dữ liệu"):
                if selected_dependent_variable:
                    st.session_state.X = st.session_state.joined_df[selected_independent_variables]
                    st.session_state.Y = st.session_state.joined_df[selected_dependent_variable]

                    # Thực hiện chia tập dữ liệu
                    X_train, X_test, Y_train, Y_test = train_test_split(st.session_state.X, st.session_state.Y, test_size=0.2, random_state=42)

                    # Lưu trữ tập dữ liệu đã chia vào session state
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.Y_train, st.session_state.Y_test = Y_train, Y_test
    
    if "Y" and "X" in st.session_state:
        if "X_train" and "Y_train" in st.session_state:
            if "X_test" and "Y_test" in st.session_state:
                colX1,colX2,colX3 = st.columns(3)
                with colX1:
                    st.write("📍Tập dữ liệu X:")
                    st.write(st.session_state.X)
                with colX2:
                    st.write("📍Tập dữ liệu X_train:")
                    st.write(st.session_state.X_train)
                with colX3:
                    st.write("📍Tập dữ liệu X_test:")
                    st.write(st.session_state.X_test)
                colY1,colY2,colY3 = st.columns(3)
                with colY1:
                    st.write("📍Tập dữ liệu Y:")
                    st.write(st.session_state.Y)
                with colY2:
                    st.write("📍Tập dữ liệu Y_train:")
                    st.write(st.session_state.Y_train)
                with colY3:
                    st.write("📍Tập dữ liệu Y_test:")
                    st.write(st.session_state.Y_test)

    # # Hiển thị tập dữ liệu đã chia trong hai cột
    #     col1, col2 = st.columns([4,3])
    #     with col1:
    #         st.subheader("Tập dữ liệu X:")
    #         st.write(st.session_state.X)
    #     with col2:
    #         st.subheader("Tập dữ liệu Y:")
    #         st.write(st.session_state.Y)

    # if "X_train" and "Y_train" in st.session_state:
    # # Hiển thị tập dữ liệu đã chia trong hai cột
    #     col1,col2 = st.columns([4,3])
        
    #     with col1:
    #         st.subheader("Tập dữ liệu X_train:")
    #         st.write(st.session_state.X_train)
    #     with col2:
    #         st.subheader("Tập dữ liệu Y_train:")
    #         st.write(st.session_state.Y_train)
    
    # if "X_test" and "Y_test" in st.session_state:
    # # Hiển thị tập dữ liệu đã chia trong hai cột
    #     col1, col2 = st.columns([4,3])
    #     with col1:
    #         st.subheader("Tập dữ liệu X_test:")
    #         st.write(st.session_state.X_test)
    #     with col2:
    #         st.subheader("Tập dữ liệu Y_test:")
    #         st.write(st.session_state.Y_test)

        # Chuẩn hóa data
    with st.sidebar.expander("Chuẩn hóa dữ liệu"):
        scaling_method = st.selectbox("Chọn phương pháp", ["None", "MinMax", "Standard", "Norm"], index=0)
        if scaling_method != "None":
            if st.button("Chuẩn hóa"):
                if "X_train" in st.session_state and "X_test" in st.session_state:
                    st.session_state.X_train_scaled, st.session_state.X_test_scaled,st.session_state.scaler  = scale_data(st.session_state.X_train, st.session_state.X_test, scaling_method)
                    st.success("Dữ liệu đã được chuẩn hóa.")

    # Hiển thị phần chuẩn hóa
                    
    if "X_train_scaled" in st.session_state and "X_test_scaled" in st.session_state:
        col_X_train, col_X_test = st.columns(2)
        with col_X_train:
            st.write("📍Tập dữ liệu X_train sau khi chuẩn hóa:")
            st.write(st.session_state.X_train_scaled)
        with col_X_test:
            st.write("📍Tập dữ liệu X_test sau khi chuẩn hóa:")
            st.write(st.session_state.X_test_scaled)
            # st.write(type(st.session_state.X_test_scaled))


    st.sidebar.title('💡Huấn luyện và đánh giá')

    with st.sidebar.expander("Chọn mô hình"):
        regression_models = ["None", "Linear Regression", "Decision Tree Regressor", "Lasso", "Ridge"]
        classification_models = ["None", "Logistic Regression", "Decision Tree Classifier", "Support Vector Machine", "Naive Bayes"]
        st.session_state.selected_regression_model = st.selectbox("Chọn mô hình hồi quy:", options=regression_models, index=0)
        st.session_state.selected_classification_model = st.selectbox("Chọn mô hình phân loại:", options=classification_models, index=0)

        if st.session_state.selected_regression_model != "None" and st.session_state.selected_classification_model != "None":
            st.warning("Vui lòng chỉ chọn một mô hình hồi quy hoặc một mô hình phân loại.")
        else:
            if st.session_state.selected_regression_model != "None":
                st.write("Mô hình hồi quy được chọn:", st.session_state.selected_regression_model)
                st.session_state.model,st.session_state.Y_pred, st.session_state.regression_metrics = train_regression_model(st.session_state.selected_regression_model, st.session_state.X_train_scaled, st.session_state.X_test_scaled, st.session_state.Y_train, st.session_state.Y_test)
                if st.session_state.regression_metrics is not None:
                    st.session_state.regression_metrics["None"] = None  # Thêm option mặc định "None"
                    st.session_state.selected_metric = st.selectbox("Chọn thông số đánh giá:", options=list(st.session_state.regression_metrics.keys()))
                    if st.session_state.selected_metric != "None":
                        if st.button("Huấn luyện"):
                            progress_bar = st.progress(0)
                            # st.write(f"Thông số đánh giá '{st.session_state.selected_metric}': {st.session_state.classification_metrics[st.session_state.selected_metric]}")
                            for percent_complete in range(100):
                                time.sleep(0.005)  # Giảm thời gian sleep để tăng tốc độ hiển thị thanh tiến trình
                                progress_bar.bar_color = 'green'
                                progress_bar.progress(percent_complete + 1,text='Đang huấn luyện dữ liệu')
                                

                            # st.write(f"Thông số đánh giá '{selected_metric}': {st.session_state.regression_metrics[selected_metric]}")
                            st.session_state.rate = st.session_state.regression_metrics[st.session_state.selected_metric]
                            st.success("Đã huấn luyện thành công.")

            elif st.session_state.selected_classification_model != "None":
                st.write("Mô hình phân loại được chọn:", st.session_state.selected_classification_model)
                st.session_state.model,st.session_state.Y_pred, st.session_state.classification_metrics = train_classification_model(st.session_state.selected_classification_model, st.session_state.X_train_scaled, st.session_state.X_test_scaled, st.session_state.Y_train, st.session_state.Y_test)
                if st.session_state.classification_metrics is not None:
                    st.session_state.classification_metrics["None"] = None  # Thêm option mặc định "None"
                    st.session_state.selected_metric = st.selectbox("Chọn thông số đánh giá:", options=list(st.session_state.classification_metrics.keys()))
                    if st.session_state.selected_metric != "None":
                        if st.button("Huấn luyện"):
                            # Hiển thị thanh tiến trình
                            progress_bar = st.progress(0)
                            # st.write(f"Thông số đánh giá '{st.session_state.selected_metric}': {st.session_state.classification_metrics[st.session_state.selected_metric]}")
                            for percent_complete in range(100):
                                time.sleep(0.005)  # Giảm thời gian sleep để tăng tốc độ hiển thị thanh tiến trình
                                progress_bar.bar_color = 'green'
                                progress_bar.progress(percent_complete + 1,text='Đang huấn luyện dữ liệu')

                            st.session_state.rate = st.session_state.classification_metrics[st.session_state.selected_metric]
                            st.success("Đã huấn luyện thành công.")
                            
            else:
                st.warning("Vui lòng chọn một mô hình để đánh giá.")
    if "rate" in st.session_state:
        st.subheader(f'Thông số đánh giá của {st.session_state.selected_metric}: {st.session_state.rate}')
        # st.write(st.session_state.rate)
    
    # if "Y_pred" and 'model' in st.session_state:
    #     st.write('Y_PRED')
    #     st.write(st.session_state.Y_pred)
    #     st.write(st.session_state.model)

    st.sidebar.title('📈Dự báo dữ liệu')
    #Dự báo mô hình
    with st.sidebar.expander("Dự đoán với dữ liệu mới"):
        # Kiểm tra xem đã có model và scaler trong session state chưa
        if "model" not in st.session_state or "scaler" not in st.session_state:
            st.warning("Vui lòng huấn luyện mô hình trước khi thực hiện dự đoán với dữ liệu mới.")
        else:
            # Lấy danh sách tên cột của dữ liệu X
            column_names = st.session_state.X_train.columns.tolist()
            
            # Tạo danh sách các ô nhập liệu cho dữ liệu mới
            st.session_state.new_data = []
            
            for column_name in column_names:
                st.session_state.new_data.append(st.number_input(f"Nhập dữ liệu cho cột '{column_name}':", value=0.0))

            

            # Kiểm tra xem tất cả các ô nhập liệu đã được điền đầy đủ chưa
            all_fields_filled = all(st.session_state.new_data)
            
            # Nếu tất cả các ô nhập liệu đã được điền đầy đủ, hiển thị nút "Dự báo"
            if all_fields_filled:
                if st.button("Dự báo"):
                    if "predictions" not in st.session_state:
                        st.session_state.predictions = []
                    st.session_state.predict_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    # Chuyển dữ liệu nhập vào thành một hàng dữ liệu mới
                    st.session_state.new_data_array = np.array(st.session_state.new_data).reshape(1, -1)
                    
                    # Chuẩn hóa dữ liệu mới bằng scaler
                    st.session_state.scaled_new_data = st.session_state.scaler.transform(st.session_state.new_data_array)

                    # Chuyển ma trận thành DataFrame
                    st.session_state.df_scaled_new_data = pd.DataFrame(data=st.session_state.scaled_new_data, columns=column_names)

                    # Dự đoán với dữ liệu mới bằng model
                    st.session_state.prediction = st.session_state.model.predict(st.session_state.df_scaled_new_data)
                    
                    # st.write(f"Nhập vào: {st.session_state.new_data}")

                    # st.write(f"Trước scaler: {st.session_state.new_data_array}")

                    # st.write(f"sau scaler: {st.session_state.scaled_new_data}")
                    # st.write(st.session_state.df_scaled_new_data)
                    # st.write(f'Type df:{type(st.session_state.df_scaled_new_data)}')
                    # Hiển thị kết quả dự đoán
                    st.write(f"Kết quả dự đoán: {st.session_state.prediction[0]}")

                    st.session_state.prediction_value = str(st.session_state.prediction[0])
                    st.session_state.data_dict = {"time": st.session_state.predict_time, "value": {"Dữ liệu nhập vào":st.session_state.new_data,'Dự Báo':st.session_state.prediction_value},}

                    # st.session_state.text_value = (f'Dữ liệu nhập vào:{st.session_state.new_data}, dữ liệu dự báo :{st.session_state.prediction[0]}')
                    # st.session_state.data_dict = {"time": st.session_state.predict_time, "value":  st.session_state.text_value}
                    st.session_state.predictions.append(st.session_state.data_dict)
            else:
                st.warning("Vui lòng nhập đầy đủ dữ liệu cho tất cả các cột.")

    if "new_data" and "new_data_array"  and "prediction" and 'scaled_new_data' and 'df_scaled_new_data' in st.session_state:
        progress_bar = st.progress(0)
        # st.write(f"Thông số đánh giá '{st.session_state.selected_metric}': {st.session_state.classification_metrics[st.session_state.selected_metric]}")
        for percent_complete in range(100):
            time.sleep(0.005)  # Giảm thời gian sleep để tăng tốc độ hiển thị thanh tiến trình
            progress_bar.bar_color = 'green'
            progress_bar.progress(percent_complete + 1,text='Đang dự đoán dữ liệu')

        st.subheader(f"Nhập vào: {st.session_state.new_data}")
        st.subheader(f"Trước scaler: {st.session_state.new_data_array}")
        st.subheader(f"Sau scaler: {st.session_state.scaled_new_data}")
        st.write(st.session_state.df_scaled_new_data)
        # Hiển thị kết quả dự đoán
        st.subheader(f"Kết quả dự đoán: {st.session_state.prediction[0]}")
        # st.write(f"Model sau dự đoán: {st.session_state.model}")

# Hàm lưu biến JSON vào file
def save_json_to_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def history():
    if "predictions" not in st.session_state:
        st.subheader("Không có dữ liệu dự báo")
    else:
        st.subheader("Lịch sử dự báo:")
        st.write(st.session_state.predictions)
        # Nút để lưu biến JSON
        if st.button("Lưu dữ liệu"):
            save_json_to_file(st.session_state.predictions, "predictions.json")
            st.success("Biến JSON đã được lưu thành công vào file 'predictions.json'.")
        for prediction in st.session_state.predictions:
            st.write(f"- {prediction['time']}: {prediction['value']}")

        
        