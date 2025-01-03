import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = load_model('plant_disease_model.h5')

# Define class names
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Create a dictionary with diseases, solutions, and fertilizers
disease_solutions = {
    'Tomato-Bacterial_spot': {
        'fertilizer': 'Use a balanced fertilizer like NPK 20-20-20 for optimal growth.',
        'detailed_solution': 'Bacterial spot in tomatoes can be treated with copper-based fungicides like copper hydroxide. Regularly remove any infected leaves and water the plants properly. Ensure proper plant spacing to improve air circulation and reduce humidity, which fosters the growth of the bacteria.'
    },
    'Potato-Barly blight': {
        'fertilizer': 'Use nitrogen-rich fertilizers like ammonium nitrate to improve plant health.',
        'detailed_solution': 'Barly blight is a fungal disease that affects potatoes. It thrives in humid conditions. Apply fungicides like chlorothalonil and remove affected plants immediately to reduce the spread of the disease. Use resistant varieties if possible to prevent recurrence.'
    },
    'Corn-Common_rust': {
        'fertilizer': 'Apply high-phosphorus fertilizers to boost the plant’s resistance.',
        'detailed_solution': 'Common rust in corn is a fungal disease that causes yellowish to orange pustules on the leaves. Use resistant varieties, apply fungicides such as tebuconazole, and remove any infected leaves to prevent the spread of the disease.'
    }
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define route to handle file upload and prediction
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template('upload.html', message="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (256, 256))  # Resize image to model's expected size
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img / 255.0  # Normalize image

            # Make prediction
            prediction = model.predict(img)
            result = CLASS_NAMES[np.argmax(prediction)]

            # Get the disease solution and fertilizer
            solution_data = disease_solutions.get(result, {
                'fertilizer': "Fertilizer information not available",
                'detailed_solution': "No detailed solution available"
            })

            return render_template('result.html', result=result, 
                                   fertilizer=solution_data['fertilizer'],
                                   detailed_solution=solution_data['detailed_solution'],
                                   image_file=filename)

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
    # Define a route to display the home page
    @app.route("/home")
    def home():
        return render_template('home.html')

    # Define a route to display information about the app
    @app.route("/about")
    def about():
        return render_template('about.html')

    # Define a route to display contact information
    @app.route("/contact")
    def contact():
        return render_template('contact.html')

    # Define a route to display the gallery of uploaded images
    @app.route("/gallery")
    def gallery():
        images = os.listdir(app.config['UPLOAD_FOLDER'])
        return render_template('gallery.html', images=images)

    # Define a route to handle file deletion
    @app.route("/delete/<filename>", methods=["POST"])
    def delete_file(filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return render_template('gallery.html', message="File deleted successfully")
        else:
            return render_template('gallery.html', message="File not found")

    # Define a route to handle file download
    @app.route("/download/<filename>")
    def download_file(filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return render_template('gallery.html', message="File not found")
