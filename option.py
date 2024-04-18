import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2


def load_df():
    df = pd.read_csv("data-before-mapping.csv")
    return df


def load_df1():
    df1 = pd.read_csv("data-before-mapping.csv")
    return df1


def load_df2():
    df2 = pd.read_csv("Data Cleaned.csv")
    return df2


def load_df3():
    df3 = pd.read_csv("train.csv")
    return df3


def scatter_plot(df):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    sns.scatterplot(x='Age', y='Purchase', data=df)
    ax.set_xlabel('Age')
    ax.set_ylabel('Purchase')
    ax.set_title("Scatter plot Age vs Purchase")
    ax.grid(True)
    st.pyplot(fig)
    text = """
    Interpretasi:
    Grafik scatterplot menunjukkan adanya korelasi positif antara usia (Age) dan nominal pembelian (Purchase). Ini mengindikasikan bahwa semakin tua usia pembeli, semakin tinggi pula nominal pembeliannya. Dengan kata lain, terdapat kecenderungan bahwa konsumen yang lebih tua cenderung melakukan pembelian dengan nominal yang lebih tinggi.

    Insight:
    Korelasi positif antara usia dan nominal pembelian menunjukkan bahwa kelompok usia tertentu, mungkin yang lebih tua, cenderung melakukan pembelian dengan nominal yang lebih besar. Hal ini dapat disebabkan oleh faktor seperti pendapatan yang lebih tinggi atau preferensi produk yang lebih mahal dari kelompok usia tersebut.

    Actionable Insight:
    Berdasarkan insight ini, supermarket dapat merancang strategi pemasaran yang lebih tepat sasaran untuk menarik konsumen dengan usia yang lebih tua. Misalnya, mereka dapat memperkenalkan paket promo khusus untuk kelompok usia tertentu atau menyediakan produk premium yang lebih sesuai dengan preferensi konsumen yang lebih tua. Dengan demikian, supermarket dapat meningkatkan penjualan dan memperluas pangsa pasar mereka di kalangan konsumen yang lebih tua.
    """
    st.write(text)


def plot_custom_correlation(df1):
    # Memperbaiki konversi nilai rentang usia menjadi nilai tunggal
    df1['Age'] = df1['Age'].apply(lambda x: (float(x.split('-')[0]) + 5) if '-' in x else (55 if '+' in x else float(x)))

    selected_columns = ['Age', 'Product_Category_1', 'Purchase']

    selected_df = df1[selected_columns]

    corr_matrix = selected_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    text = """
    Interpretasi:
    Heatmap ini memvisualisasikan korelasi antara usia pembeli, kategori produk, dan total belanja di supermarket. Warna merah menunjukkan korelasi positif yang kuat, biru menunjukkan korelasi negatif yang kuat, dan warna netral menunjukkan hubungan yang lemah atau tidak signifikan.

    Insight:
    Dari heatmap, terlihat bahwa beberapa kategori produk memiliki korelasi kuat dengan total belanja, terutama pada kelompok usia pembeli tertentu. Ini menunjukkan preferensi tertentu dari kelompok usia pembeli terhadap kategori produk tertentu, yang berpengaruh pada total belanja.

    Actionable Insight:
    Supermarket dapat menyesuaikan strategi pemasaran untuk menargetkan kelompok usia tertentu yang cenderung memiliki korelasi yang kuat dengan total belanja. Mereka dapat mengatur promosi atau menawarkan diskon khusus untuk produk-produk yang diminati oleh kelompok usia tersebut, meningkatkan penjualan dan keuntungan supermarket.
    """
    st.write(text)


def compositionAndComparison(df2):
    # Hitung rata-rata fitur untuk setiap kelas
    df2['PurchaseCategory'].replace({0: 'Category_A', 1: 'Category_B', 2: 'Category_C'}, inplace=True)
    class_composition = df2.groupby('PurchaseCategory').mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Composition for each class')
    plt.xlabel('Class')
    plt.ylabel('Feature')
    st.pyplot(plt)
    text = """
    Interpretasi:
    Visualisasi heatmap ini menampilkan rata-rata nilai fitur untuk setiap kategori pembelian. Setiap baris pada heatmap mewakili fitur tertentu, seperti Occupation (Pekerjaan), Product_Category_1 (Kategori Produk 1), Gender_F (Jenis Kelamin Wanita), dan Gender_M (Jenis Kelamin Pria). Setiap kolom mewakili kategori pembelian, yaitu Category_A (Kategori A), Category_B (Kategori B), dan Category_C (Kategori C). Warna pada heatmap menunjukkan nilai rata-rata untuk setiap fitur dan kategori pembelian. Semakin gelap warnanya, semakin tinggi nilainya. Sebaliknya, semakin terang warnanya, semakin rendah nilainya.

    Insight:
    Dari heatmap, dapat dilihat bahwa rata-rata nilai fitur Occupation untuk Category_A lebih tinggi daripada Category_B dan Category_C. Hal ini menunjukkan bahwa orang dengan pekerjaan tertentu lebih cenderung membeli produk di Category_A dibandingkan dengan kategori pembelian lainnya. 

    Actionable Insight:
    Berdasarkan insight ini, dapat dilakukan strategi pemasaran yang lebih spesifik untuk menargetkan orang dengan pekerjaan tersebut agar lebih tertarik untuk membeli produk di Category_A. Supermarket dapat mengatur promosi khusus atau menawarkan diskon untuk produk-produk di Category_A, sehingga dapat meningkatkan penjualan di kategori pembelian tersebut.
    """
    st.write(text)


def preprocess_data(df):
    # Handle missing values
    df.fillna(0, inplace=True)  # Replace missing values with 0, adjust to appropriate replacement strategy according to your data

    # Drop 'Product_ID' column
    df.drop('Product_ID', axis=1, inplace=True)

    # One-hot encoding for categorical features
    categorical_columns = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    df = pd.get_dummies(df, columns=categorical_columns)

    return df


def predict(df3):
    if df3 is not None:
        # Preprocess the data
        df3 = preprocess_data(df3)

        # Split data into features and target
        x = df3.drop('Purchase', axis=1)
        y = df3['Purchase']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Scale the features
        scaler = MinMaxScaler()
        x_train_norm = scaler.fit_transform(x_train)
        x_test_norm = scaler.transform(x_test)

        # Train Gaussian Naive Bayes model
        gnb = GaussianNB()
        gnb.fit(x_train_norm, y_train)

        # Train Decision Tree Classifier model
        dtc = DecisionTreeClassifier()
        dtc.fit(x_train_norm, y_train)

        # Predictions
        gnb_pred = gnb.predict(x_test_norm)
        dtc_pred = dtc.predict(x_test_norm)

        # Display evaluation metrics
        st.write("## Evaluation Metrics")

        st.write("### Gaussian Naive Bayes")
        gnb_accuracy = accuracy_score(y_test, gnb_pred)
        st.write("Accuracy:", gnb_accuracy)

        st.write("### Decision Tree Classifier")
        dtc_accuracy = accuracy_score(y_test, dtc_pred)
        st.write("Accuracy:", dtc_accuracy)

        # Confusion matrices
        st.write("## Confusion Matrix")
        st.write("### Gaussian Naive Bayes")
        st.write(confusion_matrix(y_test, gnb_pred))

        st.write("### Decision Tree Classifier")
        st.write(confusion_matrix(y_test, dtc_pred))

        # Classification report
        st.write("## Classification Report")
        st.write("### Gaussian Naive Bayes")
        st.write(classification_report(y_test, gnb_pred))

        st.write("### Decision Tree Classifier")
        st.write(classification_report(y_test, dtc_pred))

        # Feature ranking
        st.write("## Feature Ranking")
        selector = SelectKBest(score_func=chi2, k=10)
        selector.fit(x_train_norm, y_train)
        feature_ranks = selector.scores_
        feature_names = x_train.columns
        feature_ranks_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})
        feature_ranks_df = feature_ranks_df.sort_values(by='Rank', ascending=False)
        st.write(feature_ranks_df)

        # Describe 'Purchase' feature
        st.write("## Purchase Feature Description")
        st.write(y_test.describe())


def main():
    menu = st.sidebar.selectbox("", ["Beranda", "Distribusi", "Hubungan", "Perbandingan dan Komposisi", "Predict"])

    if menu == "Beranda":
        st.title("Analisis Data Pembelian Pelanggan")
        st.subheader("Pengaruh Umur, Gender, dan Pekerjaan terhadap Pembelian Pelanggan")
        st.write("Analisis ini bertujuan untuk mengeksplorasi hubungan antara umur, gender, dan pekerjaan terhadap pola pembelian pelanggan. Dengan memahami faktor-faktor ini, perusahaan dapat mengoptimalkan strategi pemasaran dan penjualan untuk meningkatkan kepuasan pelanggan dan hasil penjualan.")
        st.image("image/black friday.jpeg", use_column_width=True)

        st.subheader("Pengaruh Pendidikan dan Status Pernikahan terhadap Pembelian Pelanggan")
        st.write("Analisis akan menelusuri bagaimana pendidikan dan status pernikahan memengaruhi pola pembelian pelanggan. Informasi ini dapat digunakan untuk menyusun strategi pemasaran yang lebih terfokus dan menarik bagi segmen pelanggan tertentu.")
        st.image("image/black friday2.jpeg", use_column_width=True)

        st.subheader("Analisis Geografis Pembelian Pelanggan")
        st.write("Analisis akan memperlihatkan pola pembelian pelanggan berdasarkan lokasi geografis. Hal ini dapat membantu perusahaan untuk menyesuaikan strategi pemasaran dan distribusi berdasarkan preferensi konsumen di berbagai daerah.")
        st.image("image/black friday3.jpg", use_column_width=True)

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
        df2 = load_df2()  # Fixed the function call here
        compositionAndComparison(df2)

    elif menu == "Predict":
        df3 = load_df3()  # Fixed the function call here
        predict(df3)


if __name__ == '__main__':
    main()
