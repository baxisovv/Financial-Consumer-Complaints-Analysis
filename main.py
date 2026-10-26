"""
PROJE: Finansal Tüketici Şikayetlerinin Analizi (Hibrit Yaklaşım) v2.0
----------------------------------------------------------------------
Güncellemeler:
- Test Seti Oranı: %30 (test_size=0.3)
- Metrikler: Accuracy, F1-Score, Precision, Recall
- LSTM: Epoch sayısı artırıldı (10), Eğitim Grafikleri eklendi.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Makine Öğrenmesi Kütüphaneleri
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
# Derin Öğrenme Kütüphaneleri
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D


# Görselleştirme
from wordcloud import WordCloud

# --- AYARLAR ---
FILE_PATH = 'Consumer_Complaints.csv'
OUTPUT_DIR = 'görsel13'
MAX_ROWS = 140000 
RANDOM_STATE = 42
TEST_SIZE = 0.3  
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- KÜTÜPHANELER YÜKLENDİ, BAŞLIYORUZ ---")

# --- ADIM 1: NLTK HAZIRLIK VE VERİ YÜKLEME ---
def download_nltk_resources():
    print("[1/6] NLTK Kaynakları Kontrol Ediliyor...")
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except:
                pass 

download_nltk_resources()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'x{2,}', '', text)  # Anonim verileri kaldır
    text = re.sub(r'[^a-z\s]', '', text)  # Sadece harfler
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def load_and_clean_data():
    print(f"[2/6] Veri Yükleniyor: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"{FILE_PATH} bulunamadı! Lütfen dosyayı script ile aynı klasöre koyun.")
    
    df = pd.read_csv(FILE_PATH, usecols=['Consumer complaint narrative', 'Product'])
    df = df.dropna(subset=['Consumer complaint narrative'])
    
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
    
    print(f"    -> İşlenen Satır Sayısı: {len(df)}")
    print("    -> NLP Ön İşleme Başlıyor (Bu biraz sürebilir)...")
    df['cleaned_text'] = df['Consumer complaint narrative'].apply(preprocess_text)
    return df

# --- ADIM 2: K-MEANS KÜMELEME ---
def run_kmeans_analysis(df):
    print("[3/6] K-Means Kümeleme Analizi Başlıyor...")
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    
    num_clusters = 6
    kmeans = KMeans(n_clusters=num_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(tfidf_matrix)
    df['Cluster'] = kmeans.labels_
    
    # 1. Grafik: PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    scatter_plot_points = pca.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=scatter_plot_points[:,0], y=scatter_plot_points[:,1], hue=df['Cluster'], palette='viridis', s=10)
    plt.title('K-Means Kümeleme (PCA)')
    plt.savefig(f'{OUTPUT_DIR}/1_KMeans_Clusters_PCA.png')
    plt.close()
    
    # 2. Grafik: Küme Dağılımı
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Cluster', data=df, palette='viridis')
    plt.title('Şikayetlerin Kümeler Arası Dağılımı')
    plt.bar_label(ax.containers[0])
    plt.savefig(f'{OUTPUT_DIR}/2_Cluster_Distribution.png')
    plt.close()
    
    return tfidf_matrix

# --- ADIM 3: ML MODELLERİ (GÜNCELLENDİ) ---
# --- ADIM 3: ML MODELLERİ (DÜZELTİLMİŞ & CLASS WEIGHT EKLENMİŞ) ---
def run_ml_models(df, tfidf_matrix):
    print(f"[4/6] Geleneksel ML Modelleri Eğitiliyor (Test Size: {TEST_SIZE})...")
    
    le = LabelEncoder()
    df['product_id'] = le.fit_transform(df['Product'])
    
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['product_id'], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    metrics_data = [] 
    
    models = {
        'Naive Bayes': MultinomialNB(), # NB class_weight parametresi almaz, bu normal
        # GÜNCELLEME 1: Logistic Regression'a balanced eklendi
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        # GÜNCELLEME 2: Random Forest'a balanced eklendi
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    }
    
    for name, model in models.items():
        print(f"    -> {name} eğitiliyor...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        # GÜNCELLEME 3: Hem Weighted hem Macro hesaplıyoruz
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Rapor için Weighted kullanmaya devam edebilirsin ama Macro'yu da gör
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"       {name} -> Weighted F1: {f1_weighted:.3f} | Macro F1: {f1_macro:.3f}")

        metrics_data.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score (Weighted)': f1_weighted,
            'F1 Score (Macro)': f1_macro # Bunu da tabloya ekle
        })
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{OUTPUT_DIR}/3_CM_{name.replace(" ", "_")}.png')
        plt.close()
        
    return metrics_data, le

# --- ADIM 4: DERİN ÖĞRENME LSTM (GÜNCELLENDİ) ---
def run_lstm_model(df, le):
    print(f"[5/6] Derin Öğrenme (LSTM) Modeli Eğitiliyor (Class Weights Eklendi)...")
    
    MAX_NB_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 150
    EMBEDDING_DIM = 100
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['cleaned_text'].values)
    
    X = tokenizer.texts_to_sequences(df['cleaned_text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df['Product']).values
    
    # Stratify ekledik: Eğitim ve test setinde ürün dağılımı eşit olsun diye
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=Y
    )
    y_integers = np.argmax(Y_train, axis=1)
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    # Keras'ın istediği sözlük formatına çevir
    class_weight_dict = dict(enumerate(class_weights))
    print(f"    -> Sınıf Ağırlıkları Hesaplandı: {class_weight_dict}")
    # ------------------------------------------------
    
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("    -> LSTM eğitimi başlıyor (10 Epoch)...")
    history = model.fit(
        X_train, Y_train, 
        epochs=10, 
        batch_size=64, 
        validation_data=(X_test, Y_test), 
        class_weight=class_weight_dict,
        verbose=1
    )
    
    
    # LSTM Eğitim Grafikleri
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim (Train)')
    plt.plot(history.history['val_accuracy'], label='Doğrulama (Val)')
    plt.title('LSTM Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim (Train)')
    plt.plot(history.history['val_loss'], label='Doğrulama (Val)')
    plt.title('LSTM Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_LSTM_Training_History.png')
    plt.close()
    
    Y_pred_prob = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred_prob, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    
    acc = accuracy_score(Y_true, Y_pred_classes)
    # Average='weighted' yaparak dengesizliği hesaba katıyoruz
    prec = precision_score(Y_true, Y_pred_classes, average='weighted', zero_division=0)
    rec = recall_score(Y_true, Y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(Y_true, Y_pred_classes, average='weighted', zero_division=0)
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(Y_true, Y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('LSTM Confusion Matrix')
    plt.savefig(f'{OUTPUT_DIR}/4_CM_LSTM.png')
    plt.close()
    
    return {
        'Model': 'LSTM (Bizim Model)',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

# --- ADIM 5: SONUÇLARI GÖRSELLEŞTİRME VE TABLO ---
def plot_final_comparison(metrics_list):
    print("[6/6] Sonuçlar Karşılaştırılıyor...")
    
    # Sonuçları DataFrame'e çevir
    df_results = pd.DataFrame(metrics_list)
    
    # Tabloyu CSV olarak kaydet
    df_results.to_csv(f'{OUTPUT_DIR}/Final_Results_Table.csv', index=False)
    
    # Grafik: Accuracy Karşılaştırması
    plt.figure(figsize=(10, 6))
    colors = ['gray'] * (len(df_results)-1) + ['#d62728'] # LSTM Kırmızı
    bars = plt.bar(df_results['Model'], df_results['Accuracy'] * 100, color=colors)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'%{yval:.2f}', ha='center', fontweight='bold')
        
    plt.ylim(80, 100)
    plt.title('Final Model Performansı (Accuracy)')
    plt.ylabel('Doğruluk (%)')
    plt.savefig(f'{OUTPUT_DIR}/5_Final_Comparison_Accuracy.png')
    plt.close()
    
    print("\n" + "="*60)
    print("PROJE TAMAMLANDI! İŞTE DETAYLI SONUÇLAR:")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    print(f"Tüm görseller ve tablolar '{OUTPUT_DIR}' klasöründe.")

# --- ANA PROGRAM ---
if __name__ == "__main__":
    try:
        # 1. Veri
        df = load_and_clean_data()
        
        # 2. Unsupervised
        tfidf_matrix = run_kmeans_analysis(df)
        
        # 3. Supervised ML
        ml_metrics, label_encoder = run_ml_models(df, tfidf_matrix)
        
        # 4. Supervised DL (LSTM)
        lstm_metrics = run_lstm_model(df, label_encoder)
        
        # Listeleri birleştir
        ml_metrics.append(lstm_metrics)
        
        # 5. Rapor
        plot_final_comparison(ml_metrics)
        
    except Exception as e:
        print(f"\nBİR HATA OLUŞTU: {e}")