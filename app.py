import streamlit as st
from st_on_hover_tabs import on_hover_tabs  
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import tensorflow as tf
import time
from transformers import AutoTokenizer
import re
from transformers import TFAutoModelForSequenceClassification
import emoji
from keras.layers import InputLayer 
from keras.utils import custom_object_scope
import keras
from transformers import TFBertModel, TFBertMainLayer





st.set_page_config(page_title="Demo Hover Tabs", layout="wide")

st.markdown("""
    <style>
    /* Jangan ganggu main page */
    .block-container {
        padding-left: 2rem !important;  /* kasih spasi biar rapi */
        padding-right: 2rem !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }
                        
    /* sembunyikan hamburger menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* rapikan kontainer utama biar nggak kegeser */
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}

    /* style default tombol */
    .stButton > button {
        background-color: #333;
        color: white;
        border-radius: 8px;
        padding: 10px 16px;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    /* efek saat hover */
    .stButton > button:hover {
        background-color: #00cc99;
        color: black;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## Navigasi")
    tab = on_hover_tabs(
    tabName=['Home', 'Cat vs Dog', 'Food 101', 'Text Classification'],
    iconName=['home', 'pets', 'restaurant', 'text_fields'],
    styles={
        'navtab': {'background-color': '#111', 'color': '#eee', 'font-size': '16px',
                   'transition': '.3s', 'border-radius': '8px', 'padding': '8px 4px'},  
        'tabOptionsStyle': {'list-style-type': 'none', 'margin': '0', 'padding': '0'},
        'iconStyle': {'position': 'relative', 'top': '3px', 'margin-left': '4px'},  
        'tabStyle': {
            'display': 'flex',
            'align-items': 'center',
            'gap': '10px',
            'padding': '10px 8px',  
            'border-radius': '8px',
            'justify-content': 'flex-start'  
        },
        'hover': {'background-color': '#00cc99', 'color': 'black'}
    },
    default_choice=0
)


if tab == 'Home':
    st.title("🏠 Home")
    st.write("Selamat datang! Arahkan kursor ke tab di sidebar untuk berpindah halaman.")
   
    params = st.query_params
    tab = params.get("tab", "Home")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="background-color:#1e1e2f; padding:20px; border-radius:12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); min-height: 280px;">
                <h3>🐱🐶 Cat vs Dog</h3>
                <p>Memprediksi apakah gambar yang diupload adalah anjing atau kucing.</p>
                <ul>
                    <li>Upload gambar JPG/PNG/JPEG.</li>
                    <li>Model akan memproses dan menampilkan prediksi.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background-color:#1e1e2f; padding:20px; border-radius:12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); min-height: 280px;">
                <h3>🍔 Food 101</h3>
                <p>Mengenali jenis makanan dari gambar yang diupload.</p>
                <ul>
                    <li>Upload gambar makanan.</li>
                    <li>Model akan memprediksi makanan yang kamu upload.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    with col3:
          st.markdown("""
            <div style="background-color:#1e1e2f; padding:20px; border-radius:12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); min-height: 280px;">
                <h3>🔤 Text Classification</h3>
                <p>Mengklasifikasi emosi dari text yang kamu input.</p>
                <ul>
                    <li>Masukkan teks di area input.</li>
                    <li>Klik tombol jalankan untuk melihat hasil klasifikasi.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
          
    st.markdown("---")
    st.subheader("Cara menggunakan aplikasi ini:")
    st.markdown("""
    1. Pilih tab di sidebar dan pilih fitur yang diinginkan.  
    2. Ikuti instruksi di halaman ini untuk menggunakan fitur yang anda inginkan.  
    3. Tunggu proses prediksi selesai, hasil akan ditampilkan secara interaktif.  
    4. Silahkan dicoba!  
    """)
        
elif tab == 'Cat vs Dog':
    st.title("🐱🐶 Cat vs Dog")

    def preprocess(img):
        img = img.convert("RGB")
        img = img.resize((299,299))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # Model for kucing vs anjing
    model1_path = hf_hub_download(
        repo_id="Ndul/anjingvskucing",
        filename="anjingkucing.h5"
    )


    with custom_object_scope({'InputLayer': InputLayer}):
        model1 = keras.models.load_model(model1_path, compile=False, safe_mode=False)
    
    st.write("Input shape model: ", model1.input_shape)
    st.write("Upload gambar untuk mengetahui apakah itu **Kucing 🐱** atau **Anjing 🐶**.")

    # Upload gambar
    uploaded = st.file_uploader("Upload gambar", type=['jpg','jpeg','png'])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="📷 Gambar yang diupload", use_container_width=True)
        input_img = preprocess(img)

        start_time = time.time()
        
        with st.spinner("🔎 Model sedang memprediksi...", show_time=True):
            pred = model1.predict(input_img)

        duration = time.time() - start_time

        if pred.shape[1] == 1:
            label = "🐶 Anjing" if pred[0][0] > 0.5 else "🐱 Kucing"
        else:
            label = "🐶 Anjing" if np.argmax(pred[0]) == 1 else "🐱 Kucing"

        confidence = float(pred[0][0] if pred.shape[1] == 1 else pred[0][np.argmax(pred[0])]) * 100
        st.success(f"✅ Berhasil memprediksi dalam {duration:.2f} detik!.")
        st.subheader(f"Prediksi: {label}")
        st.markdown(f"**Confidence:** <span style='color:limegreen;'>{confidence:.2f}%</span>", unsafe_allow_html=True)

    else:
        st.info("Silakan upload gambar (JPG/PNG/JPEG) untuk mulai prediksi.")


elif tab == 'Food 101':
    st.title("🍽️ Food-101")

    food_emojis = {
        "apple_pie": "🥧",
        "baby_back_ribs": "🍖",
        "baklava": "🍮",
        "beef_carpaccio": "🥩",
        "beef_tartare": "🥩",
        "beet_salad": "🥗",
        "beignets": "🍩",
        "bibimbap": "🍲",
        "bread_pudding": "🍞",
        "breakfast_burrito": "🌯",
        "bruschetta": "🍞",
        "caesar_salad": "🥗",
        "cannoli": "🍰",
        "caprese_salad": "🍅",
        "carrot_cake": "🥕🍰",
        "ceviche": "🐟",
        "cheesecake": "🍰",
        "cheese_plate": "🧀",
        "chicken_curry": "🍛",
        "chicken_quesadilla": "🌮",
        "chicken_wings": "🍗",
        "chocolate_cake": "🍫🍰",
        "chocolate_mousse": "🍫🍮",
        "churros": "🥖",
        "clam_chowder": "🥣",
        "club_sandwich": "🥪",
        "crab_cakes": "🦀",
        "creme_brulee": "🍮",
        "croque_madame": "🥪🍳",
        "cup_cakes": "🧁",
        "deviled_eggs": "🥚",
        "donuts": "🍩",
        "dumplings": "🥟",
        "edamame": "🌱",
        "eggs_benedict": "🍳",
        "escargots": "🐌",
        "falafel": "🧆",
        "filet_mignon": "🥩",
        "fish_and_chips": "🐟🍟",
        "foie_gras": "🦆",
        "french_fries": "🍟",
        "french_onion_soup": "🧅🥣",
        "french_toast": "🍞🍯",
        "fried_calamari": "🦑",
        "fried_rice": "🍚",
        "frozen_yogurt": "🍦",
        "garlic_bread": "🧄🍞",
        "gnocchi": "🍝",
        "greek_salad": "🥗",
        "grilled_cheese_sandwich": "🧀🥪",
        "grilled_salmon": "🐟",
        "guacamole": "🥑",
        "gyoza": "🥟",
        "hamburger": "🍔",
        "hot_and_sour_soup": "🥣",
        "hot_dog": "🌭",
        "huevos_rancheros": "🍳🌮",
        "hummus": "🥙",
        "ice_cream": "🍨",
        "lasagna": "🍝",
        "lobster_bisque": "🦞🥣",
        "lobster_roll_sandwich": "🦞🥪",
        "macaroni_and_cheese": "🧀🍝",
        "macarons": "🍬",
        "miso_soup": "🥣",
        "mussels": "🐚",
        "nachos": "🧀🌽",
        "omelette": "🍳",
        "onion_rings": "🧅",
        "oysters": "🦪",
        "pad_thai": "🍜",
        "paella": "🥘",
        "pancakes": "🥞",
        "panna_cotta": "🍮",
        "peking_duck": "🦆",
        "pho": "🍜",
        "pizza": "🍕",
        "pork_chop": "🍖",
        "poutine": "🍟🧀",
        "prime_rib": "🥩",
        "pulled_pork_sandwich": "🥪",
        "ramen": "🍜",
        "ravioli": "🍝",
        "red_velvet_cake": "🍰",
        "risotto": "🥘",
        "samosa": "🥟",
        "sashimi": "🍣",
        "scallops": "🐚",
        "seaweed_salad": "🥗🌊",
        "shrimp_and_grits": "🍤",
        "spaghetti_bolognese": "🍝",
        "spaghetti_carbonara": "🍝🥓",
        "spring_rolls": "🥢🥟",
        "steak": "🥩",
        "strawberry_shortcake": "🍓🍰",
        "sushi": "🍣",
        "tacos": "🌮",
        "takoyaki": "🧆🐙",
        "tiramisu": "☕🍰",
        "tuna_tartare": "🐟",
        "waffles": "🧇"
    }

    food101_labels = list(food_emojis.keys())

    # Model for food101
    model1_path = hf_hub_download(
        repo_id="Ndul/food101",
        filename="food101.h5"
    )

    def preprocess_image(img):
        img = img.convert("RGB")  
        img = img.resize((299, 299))  
        img = np.array(img) / 255.0   
        img = np.expand_dims(img, axis=0)  
        return img
    

    with custom_object_scope({'InputLayer': InputLayer}):
        model3 = keras.models.load_model(model1_path, compile=False, safe_mode=False)
    
    st.write("Input shape model: ", model3.input_shape)
    st.write("Upload gambar makanan untuk mendapatkan prediksi 🍔🍕🍣")

    uploaded = st.file_uploader("Upload gambar makanan", type=['jpg','jpeg','png'])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
        input_img = preprocess_image(img)

        start_time = time.time()
        
        with st.spinner("Model sedang memprediksi...", show_time=True):
            pred = model3.predict(input_img)

        duration = time.time() - start_time

        probs = pred[0] 

        topk = st.slider("Top-K hasil yang ditampilkan", 1, 10, 5)
        
        top_indices = probs.argsort()[-topk:][::-1]
        
        st.success(f"✅ Berhasil diprediksi dalam {duration:.2f} detik!")
        st.subheader(f"🍴 {topk} Prediksi Teratas:")

        for i in top_indices:
            label = food101_labels[i].replace("_", " ").title()
            emoji = food_emojis.get(food101_labels[i], "🍽️")
            st.write(f"{emoji} **{label}** — Probabilitas: `{probs[i]:.4f}`")
    else:
        st.info("Silakan upload gambar (JPG/PNG/JPEG) untuk mulai prediksi.")
    
elif tab == 'Text Classification':
    st.title("🔤 Text Classification")
    st.info("Masukkan teks untuk mengetahui emosi yang kamu tulis.")

    st.write("📝 Ketik text untuk di klasifikasi.")
    text = st.text_area("Masukkan teks", "Halo")

    label_map = {
        0: ("😢", "SADNESS"),
        1: ("😡", "ANGER"),
        2: ("🤝", "SUPPORT"),
        3: ("🌟", "HOPE"),
        4: ("💔", "DISAPPOINTMENT")
    }
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    try:
        model2 = TFAutoModelForSequenceClassification.from_pretrained("Ndul/indobertv1")
    except Exception as e:
        st.write("from_pretrained gagal, load manual tf_model.h5:", e)
        tf_model_path = hf_hub_download(repo_id="Ndul/indobertv1", filename="tf_model.h5")

        with custom_object_scope({
            "TFBertModel": TFBertModel,
            "TFBertMainLayer": TFBertMainLayer
        }):
            model2 = tf.keras.models.load_model(tf_model_path, compile=False)
    
    slang_dict = {
        'slth': 'setelah', 'dlm': 'dalam', 'yg': 'yang', 'dg': 'dengan', 
        'hnya': 'hanya', 'jgn': 'jangan', 'skrg': 'sekarang', 'utk': 'untuk', 'presiden': 'prabowo',
        '@prabowo': 'prabowo', 'x': 'seperitnya', 'joko': 'jokowi', '@jokowi': 'jokowi', 
        '@presidenrepublikindonesia': 'prabowo', 'sbg': 'sebagai', 
    }
    emoji_dict = {
        ":smiling_face_with_heart_eyes:": "senang",
        ":fire:": "semangat",
        ":crying_face:": "sedih",
        ":thumbs_up:": "bagus",
        ":angry_face:": "marah",
        ":red_heart:": "cinta",
        ":flexed_biceps:": "semangat",   
        ":wilted_flower:": "kecewa",
        ":loudly_crying_face:": "sedih"
    }

    def preprocess_text(text):
        text = text.lower()
    
        text = emoji.demojize(text)
        for k, v in emoji_dict.items():
            text = text.replace(k, v)
        
        for slang, correct in slang_dict.items():
            text = text.replace(slang, correct)
        text = re.sub(r'^rt\s+', '', text)     
        text = re.sub(r'@\S+', '', text)       
        text = re.sub(r'http\S+', '', text)    
        text = re.sub(r'#', '', text) 
        text = re.sub(r'[^a-z\s]', '', text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    if st.button("Jalankan"):
        if text.strip() != "":
            start_time = time.time()

            clean_text = preprocess_text(text)
            enc = tokenizer(clean_text, truncation=True, padding="max_length", max_length=266, return_tensors="tf")
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            with st.spinner("Model sedang memprediksi..."):
                pred = model2(input_ids, attention_mask=attention_mask)
                logits = pred.logits.numpy()            
            
            duration = time.time() - start_time

            label_idx = int(np.argmax(logits, axis=1)[0])
            probs = tf.nn.softmax(logits, axis=1).numpy()
            
            st.success(f"✅ Berhasil memprediksi dalam {duration:.2f} detik!.")
            confidence = probs[0][label_idx] * 100
            
            emoji_icon, label_text = label_map[label_idx]
            st.subheader(f"Hasil: {emoji_icon} {label_text}")

            st.markdown(f"**Confidence:** <span style='color:limegreen;'>{confidence:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")





