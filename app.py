import streamlit as st
import numpy as np
import cv2
import time 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.sidebar.image("https://carabudidaya.co.id/wp-content/uploads/2019/05/dandelion.jpg", width=120)
with col2:
    st.sidebar.image("https://baa.itpln.ac.id/assets/front-end/img/articles/kampus_merdeka.png", width=120)
with col3:
    st.sidebar.image("https://theme.zdassets.com/theme_assets/11435355/71bfae54a7ebc3d180cfe2237cdce684505eb7e0.png", width=120)
with col4:
    st.sidebar.image("https://storage.googleapis.com/kampusmerdeka_kemdikbud_go_id/mitra/mitra_20e0600a-44d1-461e-b0eb-426361b4b836.png", width=120)

st.title("Pendeteksi Bukti Transfer")
st.header("Pendeteksi Bukti Transfer online ini memudahkan Anda untuk mengetahui bukti transaksi yang ingin dideteksi asli atau palsu")
st.write("Klik tombol 'Browse Files' dan pilih 1 file yang ingin dideteksi")
st.write("Tunggu hingga unggahan selesai dan muncul hasilnya")
from img_classification import classification
uploaded_file = st.file_uploader("Choose an image ...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR", caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classification(opencv_image, 'best_5.pt')
    #st.write(label)
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')
    for i, det in enumerate(label):
        if len(det[:, -1].unique()) == 0:
            st.success(f'Bukti transfer anda adalah asli!')
        else:
            st.success(f'Bukti transfer anda adalah palsu!')           
    
st.sidebar.header("Made By Dandelion")
text = st.sidebar.text("Nur Melini Ani | Cindi Risdayani | Fanny Maulida | Chayya Amelia Widianty | Fadjar Bachtiar Ramadhan")

