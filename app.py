import streamlit as st
import tensorflow as tf
import numpy as np
import os

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = None
model_loaded = False

try:
    if os.path.exists("dogclassification.h5"):
        model = tf.keras.models.load_model("dogclassification.h5")
        model_loaded = True
    else:
        model_loaded = False
except Exception as e:
    model_loaded = False

# ---------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------
class_names = {
    "0": "Afghan","1": "African Wild Dog","2": "Airedale","3": "American Hairless",
    "4": "American Spaniel","5": "Basenji","6": "Basset","7": "Beagle",
    "8": "Bearded Collie","9": "Bermaise","10": "Bichon Frise","11": "Blenheim",
    "12": "Bloodhound","13": "Bluetick","14": "Border Collie","15": "Borzoi",
    "16": "Boston Terrier","17": "Boxer","18": "Bull Mastiff","19": "Bull Terrier",
    "20": "Bulldog","21": "Cairn","22": "Chihuahua","23": "Chinese Crested",
    "24": "Chow","25": "Clumber","26": "Cockapoo","27": "Cocker","28": "Collie",
    "29": "Corgi","30": "Coyote","31": "Dalmation","32": "Dhole","33": "Dingo",
    "34": "Doberman","35": "Elk Hound","36": "French Bulldog","37": "German Sheperd",
    "38": "Golden Retriever","39": "Great Dane","40": "Great Perenees","41": "Greyhound",
    "42": "Groenendael","43": "Irish Spaniel","44": "Irish Wolfhound","45": "Japanese Spaniel",
    "46": "Komondor","47": "Labradoodle","48": "Labrador","49": "Lhasa",
    "50": "Malinois","51": "Maltese","52": "Mex Hairless","53": "Newfoundland",
    "54": "Pekinese","55": "Pit Bull","56": "Pomeranian","57": "Poodle","58": "Pug",
    "59": "Rhodesian","60": "Rottweiler","61": "Saint Bernard","62": "Schnauzer",
    "63": "Scotch Terrier","64": "Shar_Pei","65": "Shiba Inu","66": "Shih-Tzu",
    "67": "Siberian Husky","68": "Vizsla","69": "Yorkie"
}

# ---------------------------------------------------------
# CARE TIPS FOR EACH BREED
# ---------------------------------------------------------
care_tips = {

    "Afghan": """
    • Brush long silky coat daily  
    • Needs wide open spaces for running  
    • Gentle training recommended  
    • High-protein balanced diet required  
    """,

    "African Wild Dog": """
    • Not a domestic breed  
    • Lives only in wild habitats  
    • Highly social pack structure  
    """,

    "Airedale": """
    • Brush 2–3 times per week  
    • Very energetic – needs daily long walks  
    • Intelligent – regular training needed  
    • Check ears and coat regularly  
    """,

    "American Hairless": """
    • No shedding – great for allergies  
    • Protect from sun exposure  
    • Moisturize skin weekly  
    • Needs sweaters in cold weather  
    """,

    "American Spaniel": """
    • Brush thick coat every 2 days  
    • Prone to ear infections – clean weekly  
    • Friendly family dog  
    • Needs moderate daily exercise  
    """,

    "Basenji": """
    • Minimal shedding – brush weekly  
    • High-energy, needs daily exercise  
    • Rarely barks  
    • Independent yet clean nature  
    """,

    "Basset": """
    • Short walks due to heavy build  
    • Clean ears weekly  
    • Monitor weight carefully  
    • Very gentle, kid-friendly  
    """,

    "Beagle": """
    • Needs long daily walks  
    • Brush weekly  
    • Prone to obesity – controlled feeding  
    • Scent-driven – secure outdoor areas  
    """,

    "Bearded Collie": """
    • Brush long coat 3–4 times weekly  
    • Highly energetic and playful  
    • Needs mental challenges  
    """,

    "Bermaise": """
    • Brush dense coat daily  
    • Calm, affectionate giant  
    • Avoid heat; prefers cooler climates  
    """,

    "Bichon Frise": """
    • Hypoallergenic – groom every 4–6 weeks  
    • Playful, great indoors  
    • Needs moderate exercise  
    """,

    "Blenheim": """
    • Brush soft coat weekly  
    • Gentle toy breed  
    • Needs short walks daily  
    """,

    "Bloodhound": """
    • Clean ears weekly  
    • Long walks required  
    • Strong tracking instincts  
    • Brush weekly  
    """,

    "Bluetick": """
    • Very active – daily exercise essential  
    • Brush coat weekly  
    • Strong scent hound  
    """,

    "Border Collie": """
    • Extremely intelligent – needs mental tasks  
    • High exercise requirements  
    • Brush weekly  
    • Suitable for active owners  
    """,

    "Borzoi": """
    • Gentle, quiet, dignified  
    • Brush long coat weekly  
    • Needs open running space  
    """,

    "Boston Terrier": """
    • Sensitive to heat  
    • Minimal grooming  
    • Short walks daily  
    • Great apartment dog  
    """,

    "Boxer": """
    • Needs high physical exercise  
    • Brush short coat weekly  
    • Monitor heart and hip health  
    """,

    "Bull Mastiff": """
    • Gentle giant – low indoor activity  
    • Early obedience training necessary  
    • Brush weekly  
    """,

    "Bull Terrier": """
    • Needs daily exercise  
    • Brush weekly  
    • Strong-willed – firm training  
    """,

    "Bulldog": """
    • Clean wrinkles daily  
    • Avoid overheating  
    • Short, slow walks  
    """,

    "Cairn": """
    • Weekly brushing  
    • Needs regular playtime  
    • Curious and energetic  
    """,

    "Chihuahua": """
    • Small and fragile – handle gently  
    • Minimal exercise  
    • Keep warm in cold climates  
    """,

    "Chinese Crested": """
    • Hairless type needs sunscreen  
    • Moisturize skin regularly  
    • Light clothing for cold weather  
    """,

    "Chow": """
    • Brush thick coat 3–4 times weekly  
    • Reserved nature – early socialization needed  
    • Avoid heat  
    """,

    "Clumber": """
    • Heavy shedding – brush often  
    • Calm and gentle  
    • Short daily exercise  
    """,

    "Cockapoo": """
    • Curly coat – groom regularly  
    • Very social and affectionate  
    • Needs moderate exercise  
    """,

    "Cocker": """
    • Brush coat every 2–3 days  
    • Clean ears regularly  
    • Great family companion  
    """,

    "Collie": """
    • Brush weekly  
    • Friendly and gentle  
    • Requires regular outdoor activity  
    """,

    "Corgi": """
    • Heavy shedder – brush often  
    • Needs daily walks  
    • Watch for obesity  
    """,

    "Coyote": """
    • Wild species – not kept as a pet  
    """,

    "Dalmation": """
    • High stamina – needs running/exercise  
    • Brush short coat weekly  
    • Friendly and active  
    """,

    "Dhole": """
    • Wild species – care not applicable  
    """,

    "Dingo": """
    • Wild species – not domestic  
    """,

    "Doberman": """
    • Loyal, protective breed  
    • Needs intense daily exercise  
    • Minimal grooming  
    • Early socialization important  
    """,

    "Elk Hound": """
    • Prefers cooler climate  
    • Brush thick coat weekly  
    • Very active outdoors  
    """,

    "French Bulldog": """
    • Heat-sensitive – avoid hot weather  
    • Clean face wrinkles  
    • Short walks daily  
    """,

    "German Sheperd": """
    • Needs 1–2 hours of exercise  
    • Intelligent – loves training  
    • Brush 3–4 times weekly  
    """,

    "Golden Retriever": """
    • Brush twice weekly  
    • Needs daily play/exercise  
    • Very friendly & trainable  
    """,

    "Great Dane": """
    • Gentle and calm  
    • Needs soft bedding  
    • Moderate daily walks  
    """,

    "Great Perenees": """
    • Brush heavy coat weekly  
    • Calm guardian breed  
    • Needs space & cold-tolerant  
    """,

    "Greyhound": """
    • Low energy indoors  
    • Short, gentle coat care  
    • Enjoys brief daily runs  
    """,

    "Groenendael": """
    • Belgian shepherd – needs training  
    • Brush twice weekly  
    • Very active  
    """,

    "Irish Spaniel": """
    • Brush weekly  
    • Very affectionate  
    • Moderate daily exercise  
    """,

    "Irish Wolfhound": """
    • Gentle giant  
    • Needs space to move  
    • Brush coat weekly  
    """,

    "Japanese Spaniel": """
    • Small toy dog  
    • Clean face and eyes weekly  
    • Minimal exercise  
    """,

    "Komondor": """
    • Unique corded coat – professional grooming  
    • Natural guardian  
    • Needs space & training  
    """,

    "Labradoodle": """
    • Hypoallergenic  
    • Brush curly coat weekly  
    • Very social & friendly  
    """,

    "Labrador": """
    • Very friendly  
    • Needs daily walking  
    • Monitor weight  
    • Brush 2–3 times weekly  
    """,

    "Lhasa": """
    • Long coat – groom 3 times weekly  
    • Calm indoor breed  
    """,

    "Malinois": """
    • Police-level working breed  
    • Needs intense daily training & exercise  
    """,

    "Maltese": """
    • Brush silky coat daily  
    • Tear stain cleaning  
    • Gentle toy breed  
    """,

    "Mex Hairless": """
    • Protect skin from sun  
    • Bathe weekly  
    """,

    "Newfoundland": """
    • Excellent swimmer  
    • Brush thick coat weekly  
    • Calm and gentle  
    """,

    "Pekinese": """
    • Brush 3 times a week  
    • Monitor breathing (flat face)  
    • Avoid heat  
    """,

    "Pit Bull": """
    • Strong & athletic – daily exercise  
    • Needs socialization & training  
    • Minimal grooming  
    """,

    "Pomeranian": """
    • Brush fluffy coat 2–3 times weekly  
    • Small but active  
    """,

    "Poodle": """
    • Hypoallergenic  
    • Groom every 4–6 weeks  
    • Very intelligent  
    """,

    "Pug": """
    • Avoid heat – short nose  
    • Clean wrinkles daily  
    • Short walks only  
    """,

    "Rhodesian": """
    • Athletic hunting breed  
    • Needs large exercise area  
    """,

    "Rottweiler": """
    • Strong guardian breed  
    • Needs firm training  
    • Brush weekly  
    """,

    "Saint Bernard": """
    • Heavy shedder – brush often  
    • Avoid heat  
    • Gentle giant  
    """,

    "Schnauzer": """
    • Brush wiry coat weekly  
    • Active and smart  
    """,

    "Scotch Terrier": """
    • Brush 2–3 times weekly  
    • Independent but loyal  
    """,

    "Shar_Pei": """
    • Clean skin folds  
    • Watch for skin infections  
    • Moderate exercise  
    """,

    "Shiba Inu": """
    • Independent and clean  
    • Brush weekly  
    • Needs secure yard  
    """,

    "Shih-Tzu": """
    • Daily brushing for long coat  
    • Gentle temperament  
    • Needs regular grooming  
    """,

    "Siberian Husky": """
    • High exercise needs  
    • Heavy shedder – brush frequently  
    • Loves cold weather  
    """,

    "Vizsla": """
    • High-energy sporting dog  
    • Very affectionate  
    • Minimal grooming  
    """,

    "Yorkie": """
    • Brush silky hair daily  
    • Small, indoor-friendly  
    • Needs gentle walks  
    """
}

# Default tips if breed not in dictionary
default_care_tip = """
• Provide fresh water and quality food  
• Ensure regular exercise  
• Maintain regular vet check-ups  
• Groom depending on coat type  
• Train and socialise early  
"""

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Who's That K9?", layout="wide")

# ---------------------------------------------------------
# COMPREHENSIVE CSS STYLING (INLINE)
# ---------------------------------------------------------
st.markdown("""
<style>
/* ============================================
   GLOBAL STYLING - PALE YELLOW BACKGROUND
   ============================================ */
.stApp {
    background-color: #FFF9E6 !important;
}

/* Main container background */
.main .block-container {
    background-color: #FFF9E6 !important;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar styling */
.css-1d391kg, [data-testid="stSidebar"] {
    background-color: #FFF4D4 !important;
}

.css-1d391kg .sidebar-content {
    background-color: #FFF4D4 !important;
}

/* ============================================
   HEADINGS - ORANGE AND BOLD
   ============================================ */
h1, h2, h3, h4, h5, h6 {
    color: #FF8C00 !important;
    font-weight: 700 !important;
}

.big-title {
    font-size: 42px !important;
    font-weight: 800 !important;
    color: #FF8C00 !important;
    text-align: center !important;
    margin-bottom: 30px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.section-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #FF8C00 !important;
    margin-top: 25px !important;
    margin-bottom: 15px !important;
}

/* ============================================
   BUTTONS - ANIMATED WITH HOVER EFFECTS
   ============================================ */
.stButton > button {
    background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 40px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(255, 140, 0, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(255, 140, 0, 0.4) !important;
}

/* ============================================
   FILE UPLOADER - ROUNDED BORDER & ORANGE ACCENT
   ============================================ */
[data-testid="stFileUploader"] {
    background-color: white !important;
    border: 3px solid #FF8C00 !important;
    border-radius: 20px !important;
    padding: 25px !important;
    box-shadow: 0 4px 12px rgba(255, 140, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #FFA500 !important;
    box-shadow: 0 6px 18px rgba(255, 140, 0, 0.35) !important;
    transform: translateY(-2px);
}

.uploadedFile {
    background-color: #FFF9E6 !important;
    border-radius: 10px !important;
    border-left: 4px solid #FF8C00 !important;
}

/* ============================================
   PREDICTION RESULTS - BLACK & ORANGE THEME
   ============================================ */
.prediction-card {
    background: linear-gradient(135deg, #2C2C2C 0%, #1A1A1A 100%) !important;
    border-radius: 20px !important;
    padding: 30px !important;
    margin: 20px 0 !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
    border: 2px solid #FF8C00 !important;
    animation: fadeIn 0.8s ease-in-out;
}

.prediction-text {
    color: #FFFFFF !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin: 15px 0 !important;
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.9), 0 0 30px rgba(255, 255, 255, 0.6);
}

.confidence-text {
    color: #FFFFFF !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    text-align: center !important;
    margin: 10px 0 !important;
    text-shadow: 0 0 15px rgba(255, 255, 255, 0.9), 0 0 25px rgba(255, 255, 255, 0.6);
}

/* Success and Info boxes with black & orange theme */
.stSuccess, .stInfo {
    background: linear-gradient(135deg, #2C2C2C 0%, #1A1A1A 100%) !important;
    color: #FF8C00 !important;
    border-left: 5px solid #FF8C00 !important;
    border-radius: 15px !important;
    padding: 20px !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
    animation: fadeIn 0.8s ease-in-out;
}

.stSuccess p, .stInfo p {
    color: #FF8C00 !important;
}

/* ============================================
   CARE TIPS SECTION - BEAUTIFUL & MODERN
   ============================================ */
.care-tips-container {
    background: white !important;
    border: 3px solid #FF8C00 !important;
    border-radius: 20px !important;
    padding: 30px !important;
    margin: 25px 0 !important;
    box-shadow: 0 8px 20px rgba(255, 140, 0, 0.2) !important;
    animation: fadeIn 1s ease-in-out;
}

.care-tips-header {
    color: #FF8C00 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    margin-bottom: 20px !important;
    text-align: center;
    border-bottom: 3px solid #FF8C00;
    padding-bottom: 15px;
}

.care-tips-content {
    color: #2C2C2C !important;
    font-size: 18px !important;
    line-height: 2 !important;
    white-space: pre-wrap;
}

/* Style for markdown in care tips */
.element-container p {
    color: #2C2C2C !important;
    line-height: 1.9 !important;
}

/* ============================================
   FADE-IN ANIMATION
   ============================================ */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeIn 0.8s ease-in-out;
}

/* ============================================
   IMAGE STYLING
   ============================================ */
[data-testid="stImage"] {
    border-radius: 15px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15) !important;
    animation: fadeIn 0.8s ease-in-out;
}

/* ============================================
   RADIO BUTTONS (SIDEBAR NAVIGATION)
   ============================================ */
.stRadio > label {
    color: #2C2C2C !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

.stRadio > div {
    background-color: transparent !important;
}

/* ============================================
   ADDITIONAL STYLING
   ============================================ */
.card {
    background: white !important;
    padding: 25px !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    margin: 15px 0 !important;
    border-left: 5px solid #FF8C00 !important;
}

/* Subheader styling */
.stSubheader {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Divider */
hr {
    border-color: #FF8C00 !important;
    opacity: 0.3;
}

/* Warning message styling */
.stWarning {
    background-color: #FFF4D4 !important;
    border-left: 5px solid #FFA500 !important;
    border-radius: 10px !important;
}
/* ============================================
   ORANGE GLOW EFFECT FOR PREDICTION TEXT
   ============================================ */
.prediction-text {
    color: #FFFFFF !important;
    text-shadow:
        0 0 5px #FF8C00,
        0 0 10px #FF8C00,
        0 0 15px #FF8C00,
        0 0 20px #FFA500,
        0 0 30px #FFA500;
}

.confidence-text {
    color: #FFFFFF !important;
    text-shadow:
        0 0 5px #FF8C00,
        0 0 10px #FF8C00,
        0 0 15px #FF8C00,
        0 0 20px #FFA500,
        0 0 30px #FFA500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🐾 Who's That K9")
st.sidebar.write("Classification & Care")

page = st.sidebar.radio("Navigation", ["About Project", "Breed Prediction", "Model Metrics"])

# ---------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------
if page == "About Project":
    st.markdown('<h1 class="big-title">🐶 About the Project</h1>', unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color:#FF8C00;'>Problem Statement</h2>

    <div style="font-size:17px; line-height:1.8; color:#000000; text-align:justify;">
        Dogs are one of the earliest domesticated animals and exist with a diversity of physical and behavioural traits. Thus, we are building a robust image-based dog-breed classification system that identifies a dog's breed from a photo and returns breed-specific care information (size, lifespan, grooming, exercise, common health issues, diet tips). The system must be accurate across breeds, handle real-world photo noise (occlusion, varied lighting, poses), and integrate into a simple UI for users to upload images and receive classification plus actionable care guidance.
    </div>

    <br>

    <h2 style='color:#FF8C00;'>Proposed Solution</h2>

    <div style="font-size:17px; line-height:1.8; color:#000000;">
        <ul>
            <li><strong>Primary goal:</strong> Accurately classify dog images into breed classes using a Vision Transformer backbone fine-tuned on the dataset.</li>
            <li><strong>Secondary goal:</strong> Given a predicted breed, return a curated breed-specific care profile (textual and well-structured), such as delivering a structured care profile covering size, lifespan, grooming, exercise needs, common health issues, and diet guidance.</li>
        </ul>
    </div>

    <br>

    <h2 style='color:#FF8C00;'>Dataset Overview</h2>

    <div style="font-size:17px; line-height:1.7; color:#000000;">
        We used the <strong>Kaggle Dog Breed Identification</strong> dataset, containing:
        <ul>
            <li>~20,580 labeled dog images across <strong>120 breeds</strong></li>
            <li>Each image contains a single dog with variations in lighting, background, and pose</li>
            <li>Labels provided as CSV mapping <strong>image IDs → breed names</strong></li>
            <li>High intra-class similarity and high inter-class variance</li>
        </ul>
    </div>

    <br>

    <h2 style='color:#FF8C00;'>Methodology</h2>
    
    <h3 style='color:#FFA500; font-size:22px; font-weight:700; margin-top:20px; margin-bottom:10px;'>1️⃣ User Image Upload</h3>
    <div style="font-size:17px; line-height:1.8; color:#000000; margin-left:20px;">
        <ul>
            <li>User uploads a dog image (JPG/PNG)</li>
            <li>Image is sent to backend for classification</li>
        </ul>
    </div>

    <h3 style='color:#FFA500; font-size:22px; font-weight:700; margin-top:20px; margin-bottom:10px;'>2️⃣ Preprocessing</h3>
    <div style="font-size:17px; line-height:1.8; color:#000000; margin-left:20px;">
        <ul>
            <li>Resize image to 224 × 224 × 3</li>
            <li>Normalize (ImageNet mean/std)</li>
            <li>Optional: noise removal / background cleanup</li>
        </ul>
    </div>

    <h3 style='color:#FFA500; font-size:22px; font-weight:700; margin-top:20px; margin-bottom:10px;'>3️⃣ DeiT Feature Extraction (Pretrained Transformer)</h3>
    <div style="font-size:17px; line-height:1.8; color:#000000; margin-left:20px;">
        <ul>
            <li>Input image is converted into patch embeddings</li>
            <li>Positional encoding added</li>
            <li>DeiT Transformer extracts global visual features:
                <ul>
                    <li>Breed-specific color patterns</li>
                    <li>Body structure</li>
                    <li>Face & ear characteristics</li>
                    <li>Overall dog context</li>
                </ul>
            </li>
        </ul>
    </div>

    <h3 style='color:#FFA500; font-size:22px; font-weight:700; margin-top:20px; margin-bottom:10px;'>4️⃣ ANN Classification Head</h3>
    <div style="font-size:17px; line-height:1.8; color:#000000; margin-left:20px;">
        <ul>
            <li>ANN receives DeiT features</li>
            <li>Learns breed-specific decision boundaries</li>
            <li>Outputs: Breed + Confidence Score</li>
        </ul>
    </div>

    <h3 style='color:#FFA500; font-size:22px; font-weight:700; margin-top:20px; margin-bottom:10px;'>5️⃣ Breed Knowledge Module</h3>
    <div style="font-size:17px; line-height:1.8; color:#000000; margin-left:20px;">
        <ul>
            <li>Health Tips</li>
            <li>Grooming Tips</li>
            <li>Food & Diet Suggestions</li>
        </ul>
    </div>

    <h3 style='color:#FFA500; font-size:22px; font-weight:700; margin-top:20px; margin-bottom:10px;'>6️⃣ Final Output</h3>
    <div style="font-size:17px; line-height:1.8; color:#000000; margin-left:20px;">
        <ul>
            <li>Predicted breed</li>
            <li>Confidence score</li>
            <li>Processed dog image</li>
            <li>Care recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# BREED PREDICTION PAGE
# ---------------------------------------------------------
elif page == "Breed Prediction":
    st.markdown('<h1 class="big-title">🔍 Dog Breed Prediction</h1>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ **Model file 'dogclassification.h5' not found!**")
        st.info("""
        📁 **To enable predictions:**
        1. Upload your trained model file named `dogclassification.h5` to the project root directory
        2. Refresh the page
        
        The UI styling is complete and ready! Once you upload the model file, all prediction features will work.
        """)
        st.markdown("---")
        st.markdown("### 🎨 Preview: UI Styling Complete")
        st.markdown("""
        ✅ Pale yellow background  
        ✅ Animated orange buttons  
        ✅ Black & orange themed results  
        ✅ Beautiful rounded cards with shadows  
        ✅ Modern care tips section  
        ✅ Styled file uploader  
        ✅ Fade-in animations  
        """)
    else:
        st.markdown('<h2 class="section-title">Upload Image</h2>', unsafe_allow_html=True)

        uploaded_image = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            img = tf.image.decode_image(uploaded_image.read(), channels=3)
            img = tf.image.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0) / 255.0

        if st.button("🐕 Predict Breed"):
            prediction = model.predict(img)

            pred_index = np.argmax(prediction)
            predicted_class = class_names[str(pred_index)]

            confidence = float(np.max(prediction)) * 100

            # --------- CONFIDENCE THRESHOLD LOGIC ----------
            if confidence < 35:
                st.markdown(f"""
                <div class="prediction-card">
                    <p class="prediction-text">⚠️ Not a Dog or it is not a clear picture.Upload Another </p>
                    <p class="confidence-text">📊 Confidence Score: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Clear breed so care-tips DO NOT appear
                if "predicted_breed" in st.session_state:
                    del st.session_state["predicted_breed"]

            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <p class="prediction-text">🎯 Predicted Breed: {predicted_class}</p>
                    <p class="confidence-text">📊 Confidence Score: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.session_state["predicted_breed"] = predicted_class


    if "predicted_breed" in st.session_state:
        breed = st.session_state["predicted_breed"]

        st.markdown('<br>', unsafe_allow_html=True)
        
        tips = care_tips.get(breed, default_care_tip)

        st.markdown(f"""
        <div class="care-tips-container">
            <h2 class="care-tips-header">🌟 Care Tips for {breed}</h2>
            <div class="care-tips-content">{tips}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL METRICS PAGE
# ---------------------------------------------------------
elif page == "Model Metrics":
    st.markdown('<h1 class="big-title">📈 Model Metrics</h1>', unsafe_allow_html=True)

    st.subheader("Overall Performance Metrics")

    training_acc = 0.9217
    validation_acc = 0.9056

    st.markdown(f"""
    **Training Accuracy:** {training_acc*100:.2f}%  
    **Validation Accuracy:** {validation_acc*100:.2f}%  

    """)

    st.markdown("---")

    st.subheader("Training vs Validation Loss")

    st.image("assests2/Training_and_validation_loss.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    • Training and validation loss both decrease smoothly.  
    • No major gap between them → indicates **good generalization**.  
    • Model is learning progressively without overfitting.
    """)

    st.markdown("---")

    st.subheader("Training vs Validation Accuracy")

    st.image("assests2/Training_and_validation_accuracy.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    • Accuracy climbs rapidly during early epochs.  
    • Validation accuracy stays close to training accuracy → **stable performance**.  
    • Final validation accuracy reaches ~90.56%.
    """)

    st.markdown("---")

    st.subheader("Confidence Distribution on Test Images")

    st.image("assests2/Confidence_scores_for_test_images.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    • Majority of predictions have **high confidence (70–95%)**.  
    • A few lower-confidence bars indicate challenging or ambiguous images.  
    • The model is generally confident and consistent.
    """)

    st.markdown("---")

    st.subheader("Model Attention Visualization")

    st.image("assests2/attention.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    • **Original Image:** Shows the input dog image.  
    • **Attention Heatmap:** Red areas indicate where the model focuses most during prediction.  
    • **Attention Overlay:** Combines both to visualize which parts of the dog (face, ears, body) the model considers important for breed classification.  
    • The model correctly focuses on breed-discriminative features.
    """)
