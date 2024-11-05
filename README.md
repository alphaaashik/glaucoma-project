# glaucoma-project
This project is consits of glaucoma prediction using cnn model and glaucoma types prediction using the dnn model.
#REQUIREMENTS
Flask==2.2.2
Flask-Cors==3.0.10
Flask-Mail==0.9.1
torch==1.13.1
torchvision==0.14.1
Pillow==9.3.0
pandas==1.5.1
numpy==1.23.4
scikit-learn==1.1.3
tensorflow==2.12.0
joblib==1.2.0
#templates
#home.html
This is the first page of the project it conists of details about the glaucoma and highlights of the features of this project
#predict.html
This Glaucoma Prediction page allows users to enter personal details and medical history to assess the likelihood of glaucoma. Based on the provided information, users receive predictions and recommended medications tailored to their specific glaucoma type.
#image_predict.html
Upload an image to determine if it exhibits signs of Glaucoma or is classified as Non-Glaucoma. Get instant results to assist in your eye health evaluation.
#treatment.html
Explore the various stages of glaucoma, from early detection to advanced cases, and understand the corresponding treatment options. This guide provides essential information to help you manage and respond to glaucoma effectively.
#contact.html
Reach out to us through the Contact Us page to share your inquiries or feedback. Fill out the form with your name, email, and message, and our team will get back to you promptly.
#static
#styles.css
It consists of the overall design of the project.
#app1.py
It is used for the backend running the code of the models and backend
#model.pth
It is the cnn model for prediction glauocma using image.
#dnn_model1.h5
It is used for glaucoma types.
