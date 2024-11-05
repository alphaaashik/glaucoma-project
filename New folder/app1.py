from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_cors import CORS
from flask_mail import Mail, Message  # Import Flask-Mail
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import joblib
import os

# Generate a secret key using os.urandom
secret_key = os.urandom(24)  # Generates 24 random bytes
# print(secret_key.hex())  # Converts to a hex string

app = Flask(__name__)
CORS(app)  # Enable CORS to handle requests from different origins

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Example for Gmail
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'projects3024@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'eczx gujn gwuw gbie'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'projects3024@gmail.com'
app.secret_key = secret_key.hex()   # Replace with your secret key for session management

mail = Mail(app)
# Load the EfficientNet model for image classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth', map_location=device)
model = model.to(device)
model.eval()

# Image transformations for EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the DNN model, scaler, and label encoder for glaucoma type prediction
dnn_model = load_model('dnn_model1.h5')
scaler = joblib.load('scaler1.pkl')
label_encoder = joblib.load('label_encoder1.pkl')
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

# List of features that the model was trained on
trained_feature_columns = [
    'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 
    'Pachymetry', 'LogMAR VA', 'VFT Sensitivity', 'VFT Specificity', 
    'OCT RNFL Thickness (µm)', 'OCT GCC Thickness (µm)', 
    'OCT Retinal Volume (mm³)', 'OCT Macular Thickness (µm)', 
    'Amoxicillin', 'Aspirin', 'Atorvastatin', 'Ibuprofen', 
    'Lisinopril', 'Metformin', 'Omeprazole', 'Blurred vision', 
    'Eye pain', 'Halos around lights', 'Nausea', 'Redness in the eye', 
    'Tunnel vision', 'Vision loss', 'Vomiting', 'Gender_Male', 
    'Family History_Yes', 'Medical History_Glaucoma in family', 
    'Medical History_Hypertension', 'Cataract Status_Present', 
    'Angle Closure Status_Open'
]
# Dictionary for glaucoma types and their medications
glaucoma_medications = {
    "Juvenile Glaucoma": [
        "Prostaglandin analogs (e.g., Latanoprost, Bimatoprost)",
        "Beta-blockers (e.g., Timolol)",
        "Carbonic anhydrase inhibitors (e.g., Dorzolamide, Brinzolamide)",
        "Alpha agonists (e.g., Brimonidine)"
    ],
    "Normal-Tension Glaucoma": [
        "Prostaglandin analogs (e.g., Latanoprost, Bimatoprost)",
        "Beta-blockers (e.g., Timolol, Betaxolol)",
        "Carbonic anhydrase inhibitors (e.g., Dorzolamide, Acetazolamide)",
        "Alpha agonists (e.g., Brimonidine)"
    ],
    "Secondary Glaucoma": [
        "Medications depend on the underlying cause",
        "Prostaglandin analogs (e.g., Latanoprost, Bimatoprost)",
        "Beta-blockers (e.g., Timolol)",
        "Carbonic anhydrase inhibitors (e.g., Dorzolamide, Brinzolamide)"
    ],
    "Primary Open-Angle Glaucoma": [
        "Prostaglandin analogs (e.g., Latanoprost, Travoprost)",
        "Beta-blockers (e.g., Timolol, Betaxolol)",
        "Carbonic anhydrase inhibitors (e.g., Dorzolamide, Brinzolamide)",
        "Rho kinase inhibitors (e.g., Netarsudil)"
    ],
    "Angle-Closure Glaucoma": [
        "Carbonic anhydrase inhibitors (e.g., Acetazolamide)",
        "Osmotic agents (e.g., Mannitol)",
        "Pilocarpine (to induce miosis)",
        "Beta-blockers (e.g., Timolol)"
    ],
    "Congenital Glaucoma": [
        "Prostaglandin analogs (e.g., Latanoprost)",
        "Beta-blockers (e.g., Timolol)",
        "Carbonic anhydrase inhibitors (e.g., Dorzolamide)",
        "Surgical intervention may be necessary in severe cases"
    ]
}

def preprocess_input_data(input_data):
    """Preprocesses input data for the model."""
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([{
        'Visual Acuity Measurements': input_data.get('Visual Acuity Measurements', ''),
        'VFT Sensitivity': float(input_data.get('VFT Sensitivity', 0)),
        'VFT Specificity': float(input_data.get('VFT Specificity', 0)),
        'Age': float(input_data.get('age', 0)),
        'Intraocular Pressure (IOP)': float(input_data.get('eye-pressure', 0)),
        'Cup-to-Disc Ratio (CDR)': 0,  # Placeholder, set to relevant value if available
        'Pachymetry': 0,                # Placeholder, set to relevant value if available
        'OCT RNFL Thickness (µm)': 0,   # Placeholder, set to relevant value if available
        'OCT GCC Thickness (µm)': 0,     # Placeholder, set to relevant value if available
        'OCT Retinal Volume (mm³)': 0,    # Placeholder, set to relevant value if available
        'OCT Macular Thickness (µm)': 0   # Placeholder, set to relevant value if available
    }])

    # Initialize missing columns with default values
    for col in trained_feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Map 'Visual Acuity Measurements' to 'LogMAR VA'
    input_df['LogMAR VA'] = input_df['Visual Acuity Measurements'].apply(
        lambda x: float(x.split()[1]) if 'LogMAR' in x else (0.3 if x == '20/40' else (0.0 if x == '20/20' else np.nan))
    )

    # Setting values for medication usage based on input
    input_df['Amoxicillin'] = 1 if input_data.get('Amoxicillin', False) else 0
    input_df['Ibuprofen'] = 1 if input_data.get('Ibuprofen', False) else 0
    input_df['Atorvastatin'] = 1 if input_data.get('Atorvastatin', False) else 0
    input_df['Aspirin'] = 1 if input_data.get('Aspirin', False) else 0
    input_df['Omeprazole'] = 1 if input_data.get('Omeprazole', False) else 0
    input_df['Lisinopril'] = 1 if input_data.get('Lisinopril', False) else 0
    input_df['Metformin'] = 1 if input_data.get('Metformin', False) else 0

    # Map visual symptoms
    symptoms = input_data.get('Visual Symptoms', [])
    input_df['Tunnel vision'] = 1 if 'Tunnel vision' in symptoms else 0
    input_df['Vomiting'] = 1 if 'Vomiting' in symptoms else 0
    input_df['Redness in the eye'] = 1 if 'Redness in the eye' in symptoms else 0
    input_df['Nausea'] = 1 if 'Nausea' in symptoms else 0
    input_df['Eye pain'] = 1 if 'Eye pain' in symptoms else 0
    input_df['Halos around lights'] = 1 if 'Halos around lights' in symptoms else 0
    input_df['Blurred vision'] = 1 if 'Blurred vision' in symptoms else 0
    input_df['Vision loss'] = 1 if 'Vision loss' in symptoms else 0

    # Map gender
    input_df['Gender_Male'] = 1 if input_data.get('Gender', '').lower() == 'male' else 0

    # Map family history
    input_df['Family History_Yes'] = 1 if input_data.get('Family History', '').lower() == 'yes' else 0

    # Map medical history
    medical_history = input_data.get('Medical History', '').lower()
    input_df['Medical History_Glaucoma in family'] = 1 if medical_history == 'glaucoma in family' else 0
    input_df['Medical History_Hypertension'] = 1 if medical_history == 'hypertension' else 0

    # Map cataract status
    input_df['Cataract Status_Present'] = 1 if input_data.get('Cataract Status', '').lower() == 'present' else 0

    # Map angle closure status
    input_df['Angle Closure Status_Open'] = 1 if input_data.get('Angle Closure Status', '').lower() == 'open' else 0

    # Keep only the trained features
    input_df = input_df[trained_feature_columns]

    # Ensure input has the correct number of features
    if input_df.shape[1] != len(trained_feature_columns):
        raise ValueError(f"Expected {len(trained_feature_columns)} input features but got {input_df.shape[1]}.")

    # Print the processed columns for debugging
    print("Processed input columns:", input_df.columns.tolist())

    # Check the shape of the processed input
    print("Input shape before model:", input_df.shape)

    # Return the processed input as a numpy array for the model
    return input_df.values



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/image-predict')
def image_prediction_page():
    return render_template('image_predict.html')

@app.route('/treatment')
def treatment_page():
    return render_template('treatment.html')

# @app.route('/contact', methods=['GET', 'POST'])

@app.route('/contact', methods=['GET', 'POST'])
def contact_page():
    """Contact form for users to send messages."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Create the email message with user's email in the body
        msg = Message(subject='New Contact Form Submission',
                      recipients=['projects3024@gmail.com'],  # Replace with your email
                      body=f'From: {name} <{email}>\n\n{message}')

        try:
            mail.send(msg)
            flash('Message sent successfully!', 'success')
        except Exception as e:
            flash(f'Message could not be sent. Error: {str(e)}', 'error')  # Provide error details

        return redirect(url_for('contact_page'))

    return render_template('contact.html')
@app.route('/about')
def about_page():
    return render_template('about.html')
# def contact_page():
#     """Contact form for users to send messages."""
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         message = request.form['message']

#         msg = Message(subject='New Contact Form Submission',
#                       recipients=['projects3024@gmail.com'],  # Replace with your email
#                       body=f'From: {name} <{email}>\n\n{message}')

#         try:
#             mail.send(msg)
#             flash('Message sent successfully!', 'success')
#         except Exception as e:
#             flash(f'Message could not be sent. Error: {str(e)}', 'error')  # Provide error details

#         return redirect(url_for('contact_page'))

#     return render_template('contact.html')

# @app.route('/contact')
# def contact_page():
#     return render_template('contact.html')

@app.route('/image-predict', methods=['POST'])
def predict_image_classification():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
        try:
            image = Image.open(file).convert('RGB')
            predicted_class = classify_image(image)
            class_labels = {0: 'Glaucoma', 1: 'Non-Glaucoma'}
            return jsonify({'prediction': class_labels[predicted_class]})
        except Exception as e:
            return jsonify({'error': f'Error during image prediction: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file format. Please upload a .jpg, .jpeg, or .png image.'}), 400

def predict_glaucoma(input_data):
    input_scaled = preprocess_input_data(input_data)
    predictions = dnn_model.predict(input_scaled)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = label_encoder.inverse_transform(predicted_class)
    return class_labels[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        print("Received input data:", input_data)
        input_scaled = preprocess_input_data(input_data)

        predictions = dnn_model.predict(input_scaled)
        predicted_class = np.argmax(predictions, axis=1)
        class_labels = label_encoder.inverse_transform(predicted_class)

        return jsonify({'prediction': class_labels[0]})
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500

def classify_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == '__main__':
    app.run(debug=True)
