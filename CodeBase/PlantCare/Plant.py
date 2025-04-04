import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Light UI Theme
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: #2563eb;
        color: white;
        font-weight: 500;
        border: none;
        margin: 10px 0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .stButton>button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
    }
    
    /* Upload Box Styles */
    .upload-box {
        border: 2px dashed #e5e7eb;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
        background-color: #f9fafb;
        transition: all 0.2s ease;
    }
    .upload-box:hover {
        border-color: #2563eb;
        background-color: #f3f4f6;
    }
    
    /* Result Box Styles */
    .success-box {
        background: #f8fafc;
        padding: 24px;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin: 20px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Container Styles */
    .header-container {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #f1f5f9;
    }
    
    /* Feature Card Styles */
    .feature-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 12px 0;
        transition: all 0.2s ease;
        border: 1px solid #f1f5f9;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Typography Styles */
    h1 {
        color: #1e293b;
        font-size: 2.5em !important;
        font-weight: 600 !important;
        margin-bottom: 0.5em !important;
        letter-spacing: -0.5px;
        font-family: 'Inter', sans-serif;
    }
    h3 {
        color: #334155 !important;
        font-size: 1.5em !important;
        font-weight: 500 !important;
        letter-spacing: -0.3px;
        font-family: 'Inter', sans-serif;
    }
    p {
        color: #64748b !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #f1f5f9;
    }
    
    /* Model Selection Styles */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 8px;
        width: 100% !important;  /* Make dropdown wider */
    }
    
    /* Progress Bar Styles */
    .stProgress > div > div > div > div {
        background-color: #2563eb;
    }
    
    /* Image Display Styles */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Spinner Styles */
    .stSpinner > div {
        border-color: #2563eb !important;
    }

    /* Link Styles */
    a {
        color: #2563eb !important;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }
    a:hover {
        color: #1d4ed8 !important;
    }

    /* Sidebar Navigation */
    .css-1d391kg .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #f8fafc;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f8fafc;
    }
    </style>
    
    <style>
    /* Fix dropdown text visibility */
    .stSelectbox > div {
        font-size: 16px !important;
        color: black !important;
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)

def load_model_2(model_name):
    try:
        if model_name == 'ResNet50':
            model = tf.keras.models.load_model("plant-village_disease_model_resnet50.keras")
            target_size = (224, 224)
        return model, target_size
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    
# Function to load the model based on the user's selection
def load_model(model_name, dataset_name):
    try:
        # Dataset specific models
        if dataset_name == 'new plant village':
            if model_name == 'CNN':
                model = tf.keras.models.load_model("trained_plant_disease_model.keras")
                target_size = (128, 128)
            elif model_name == 'ResNet50':
                model = tf.keras.models.load_model("trained_plant_disease_model_resnet50.keras")
                target_size = (224, 224)
            elif model_name == 'VGG16':
                model = tf.keras.models.load_model("trained_plant_disease_model_vgg16.keras")
                target_size = (224, 224)
            elif model_name == 'Alexnet':
                model = tf.keras.models.load_model("trained_plant_disease_alexnet_model.keras")
                target_size = (128, 128)
            elif model_name == 'DenseNet121':
                model = tf.keras.models.load_model("trained_plant_disease_model_densenet121.keras")
                target_size = (224, 224)
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                          'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
                          'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy',
                          'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                          'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
                          'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
                          'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
                          'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
                          'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
                          'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites_Two-spotted_spider_mite',
                          'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                          'Tomato___healthy']
        
        elif dataset_name == 'plant village':
            if model_name == 'plantvillage_densenet':
                model = tf.keras.models.load_model("densenet_plant_disease_model.keras")
                target_size = (128, 128)
            elif model_name == 'plantvillage_efficientnet':
                model = tf.keras.models.load_model("efficientnet_plant_disease_model_final.keras")
                target_size = (128, 128)
            elif model_name == 'plantvillage_resnet50':
                model = tf.keras.models.load_model("plant-village_disease_model_resnet50.keras")
                target_size = (224, 224)
            class_name = ['Apple scab', 'Apple rust', 'Apple healthy','Blueberry healthy', 'Cherry Powdery mildew', 'Cherry healthy',
                          'Corn Cercospora leaf spot', 'Corn Common rust', 'Corn Northern leaf blight', 'Corn healthy',
                          'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy',
                          'Orange Haunglongbing', 'Peach Bacterial spot', 'Peach healthy',
                          'Pepper Bacterial spot', 'Pepper healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy',
                          'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy',
                          'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf mold', 'Tomato Septoria leaf spot',
                          'Tomato Spider mites', 'Tomato Target spot', 'Tomato Yellow leaf curl virus', 'Tomato Mosaic virus', 'Tomato healthy']
        
        return model, target_size, class_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# TensorFlow Model Prediction
def model_prediction(test_image, model, target_size):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

def generate_visualization(test_image, model):
    img = image.load_img(test_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer('conv5_block3_3_conv')
    grad_model = Model(inputs=[base_model.input], outputs=[last_conv_layer.output, base_model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_class]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i].numpy()
    
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    
    plt.imshow(img)
    plt.imshow(heatmap, alpha=0.6, cmap='jet')
    plt.axis('off')
    
    return plt


# Sidebar with dataset and model selection
with st.sidebar:
    st.title("🌱 PlantCare")
    st.markdown("---")
    app_mode = st.selectbox(
        "Navigation",
        ["Home", "About", "Disease Recognition","Visualisation", "Accuracy Comparison"],
        format_func=lambda x: {
            "Home": "🏠 Home",
            "About": "ℹ️ About",
            "Disease Recognition": "🔍 Disease Recognition",
            "Visualisation":"📊 Visualisation",
            "Accuracy Comparison": "📈 Accuracy Comparison"
        }[x]
    )

# Main content
if app_mode == "Home":
    st.markdown("""
        <div class="header-container">
            <h1>Plant Disease Recognition</h1>
            <p style="font-size: 1.2em; color: #64748b;">
                Advanced ML-powered plant health monitoring system
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main image with caption
    st.image("home_page.jpeg", use_container_width=True, caption="AI-powered plant disease detection")

    # Key Features
    st.markdown("### Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>🎯 Precise Detection</h3>
                <p>State-of-the-art AI models for accurate disease identification</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>⚡ Real-time Analysis</h3>
                <p>Instant results for quick decision making</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🔄 Multi-model System</h3>
                <p>Multiple AI models for comprehensive analysis</p>
            </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("### Process")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>1. Upload</h3>
                <p>Submit a clear image of the plant</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>2. Analyze</h3>
                <p>AI processes the image data</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>3. Results</h3>
                <p>Receive detailed analysis report</p>
            </div>
        """, unsafe_allow_html=True)
        
elif app_mode == "Accuracy Comparison":
    st.title("Model Accuracy Comparison")

    # Data
    datasets = {
        "New Plant Diseases": pd.DataFrame({
            "Model": ["CNN", "AlexNet", "EfficientNet", "ResNet", "MobileNetV2"],
            "Accuracy": [95.94, 93.74, 4.44, 99.07, 92.12]
        }),
        "Plant Village": pd.DataFrame({
            "Model": ["AlexNet", "EfficientNet", "VGG16", "ResNet50", "InceptionV3", "VanillaCNN", "DenseNet121"],
            "Accuracy": [96.71, 99.40, 95.46, 96.19, 18.07, 95.69, 47.00]
        })
    }

    # Dropdown for dataset selection
    selected_dataset = st.selectbox("Select a Dataset", list(datasets.keys()))

    # Get the selected dataset
    df = datasets[selected_dataset]

    # Create Plotly chart
    fig = px.bar(df, x="Model", y="Accuracy", text=df["Accuracy"], color="Accuracy",
                 color_continuous_scale="viridis", title=f"Model Accuracy Comparison ({selected_dataset})")

    # Adjust layout
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_yaxes(range=[0, 100], title="Accuracy (%)")
    fig.update_xaxes(title="Model")

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif app_mode == "About":
    st.markdown("""
        <div class="header-container">
            <h1>About the System</h1>
            <p style="font-size: 1.2em; color: #64748b;">
                Leveraging ML for precise plant disease detection
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Dataset Information
    st.markdown("""
        <div class="feature-card">
            <h3>📊 Dataset Overview - New Plant Village</h3>
            <p>Comprehensive analysis based on over 87,000 plant images for accurate disease detection.</p>
        </div>
    """, unsafe_allow_html=True)
    
     # Dataset Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>70,295</h3> 
                <p>Training Images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>17,572</h3>
                <p>Validation Images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>38</h3>
                <p>Plant Categories</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <h3>📊 Dataset Overview - Plant Village</h3>
            <p>Comprehensive analysis based on over 70,000 plant images for accurate disease detection.</p>
        </div>
    """, unsafe_allow_html=True)
    
     # Dataset Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>53,690</h3> 
                <p>Training Images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>17,572</h3>
                <p>Validation Images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>38</h3>
                <p>Plant Categories</p>
            </div>
        """, unsafe_allow_html=True)

   

elif app_mode == "Disease Recognition":
    st.markdown("""
        <div class="header-container">
            <h1>Disease Analysis</h1>
            <p style="font-size: 1.2em; color: #64748b;">
                Upload an image for instant disease detection
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Dropdown for selecting the dataset
    dataset_option = st.selectbox("Select Dataset", ["new plant village", "plant village"])
    
    # Dropdown for selecting the model based on the dataset
    if dataset_option == "new plant village":
        model_option = st.selectbox("Select Model", ["CNN", "ResNet50", "VGG16", "Alexnet", "DenseNet121"])
    elif dataset_option == "plant village":
        model_option = st.selectbox("Select Model", ["plantvillage_densenet", "plantvillage_efficientnet", "plantvillage_resnet50"])

    # Load the model
    model, target_size, class_name = load_model(model_option, dataset_option)

    # File upload
    test_image = st.file_uploader("Upload plant image", type=['jpg', 'jpeg', 'png'])

    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="feature-card">
                    <h3>📷 Uploaded Image</h3>
                </div>
            """, unsafe_allow_html=True)
            st.image(test_image, use_container_width=True)

        with col2:
            st.markdown("""
                <div class="feature-card">
                    <h3>📊 Analysis Results</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Processing image..."):
                    result_index = model_prediction(test_image, model, target_size)
                    
                    st.markdown(f"""
                        <div class="success-box">
                            <h3>Analysis Results</h3>
                            <p style="font-size: 1.2em; color: #334155 !important; font-weight: 500;">
                                Detected Condition: {class_name[result_index]}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
elif app_mode == "Visualisation":
    
    st.markdown("<h1>Visualisation</h1>", unsafe_allow_html=True)
    model_option = st.selectbox("Select Model", ["ResNet50"], help="Choose AI model")
    model, target_size = load_model_2(model_option)
    test_image = st.file_uploader("Upload plant image", type=['jpg', 'jpeg', 'png'])
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(test_image, use_container_width=True)
        with col2:
            if st.button("Analyze Image"):
                with st.spinner("Processing image..."):
                    result_index = model_prediction(test_image, model, target_size)
                class_name = ['Apple scab', 'Apple rust', 'Apple healthy','Blueberry healthy', 
                                'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot', 
                                'Corn Common rust', 'Corn Northern leaf blight', 'Corn healthy',
                                'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy',
                                'Orange Haunglongbing', 'Peach Bacterial spot', 'Peach healthy',
                                'Pepper Bacterial spot', 'Pepper healthy', 'Potato Early blight', 
                                'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 
                                'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 
                                'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 
                                'Tomato Late blight', 'Tomato Leaf mold', 'Tomato Septoria leaf spot',
                                'Tomato Spider mites', 'Tomato Target spot', 
                                'Tomato Yellow leaf curl virus', 'Tomato Mosaic virus', 'Tomato healthy']
                
                 # Replace with actual class names
                # st.success(f"Detected Condition: {class_name[result_index]}")
                st.write("### Visualization")
                vis_plot = generate_visualization(test_image, model)
                st.pyplot(vis_plot)
