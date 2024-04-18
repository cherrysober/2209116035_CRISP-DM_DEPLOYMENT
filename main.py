import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from streamlit_option_menu import *
from option import *

df = pd.read_csv('data-before-mapping.csv')
df1 = pd.read_csv('data-before-mapping.csv')
df2 = pd.read_csv('Data Cleaned.csv')
df3 = pd.read_csv('train.csv')


def main():
    menu = st.sidebar.selectbox("", ["Beranda", "Distribusi", "Hubungan", "Perbandingan dan Komposisi", "Predict"])
    
    if menu == "Beranda":
        st.image("https://img.freepik.com/premium-vector/black-friday-sale-shopping-vector-illustration_710940-101.jpg")
        st.title("Analisis Data Pembelian Pelanggan")
        st.subheader("Pengaruh Umur, Gender, dan Pekerjaan terhadap Pembelian Pelanggan")
        st.write("Analisis ini bertujuan untuk mengeksplorasi hubungan antara umur, gender, dan pekerjaan terhadap pola pembelian pelanggan. Dengan memahami faktor-faktor ini, perusahaan dapat mengoptimalkan strategi pemasaran dan penjualan untuk meningkatkan kepuasan pelanggan dan hasil penjualan.")
        
        
        st.subheader("Pengaruh Pendidikan dan Status Pernikahan terhadap Pembelian Pelanggan")
        st.write("Analisis akan menelusuri bagaimana pendidikan dan status pernikahan memengaruhi pola pembelian pelanggan. Informasi ini dapat digunakan untuk menyusun strategi pemasaran yang lebih terfokus dan menarik bagi segmen pelanggan tertentu.")
        

        st.subheader("Analisis Geografis Pembelian Pelanggan")
        st.write("Analisis akan memperlihatkan pola pembelian pelanggan berdasarkan lokasi geografis. Hal ini dapat membantu perusahaan untuk menyesuaikan strategi pemasaran dan distribusi berdasarkan preferensi konsumen di berbagai daerah.")
        

    elif menu == "Distribusi":
        st.title("Data Distribusi")
        df = load_df()
        scatter_plot(df)

    elif menu == "Hubungan":
        st.title("Hubungan")
        df1 = load_df1()
        plot_custom_correlation(df1)
    
    elif menu == "Perbandingan dan Komposisi":
        st.title('Composition')
        compositionAndComparison(df2)
    
    elif menu == "Predict":
        predict(df3)    

if __name__ == '__main__':
    main()
