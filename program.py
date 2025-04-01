import tensorflow as tf
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from flask import Flask, request, send_from_directory
import os

# Define dataset paths
base_dir = "data"
train_dir = f"{base_dir}/train"

# Check if model exists, otherwise train it
model_path = "models/dog_cat_classifier.keras"
if os.path.exists(model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print("Loaded existing model from disk")
else:
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Data preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, 
        rotation_range=20, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        horizontal_flip=True, 
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',  
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Define and train the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    
    # Save the model after training
    model.save(model_path, save_format='keras')
    print(f"Model saved to {model_path}")

# Flask setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.static_folder = "static"

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file has been uploaded!"
        
        file = request.files["file"]
        if file.filename == "":
            return "No file has been chosen!"
        
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        
        prediction = predict_image(file_path)
        
        return f'''
        <!doctype html>
        <html lang="sv">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dog eller Cat?</title>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <h1>Result</h1>
            <div class="result">
                <h2>{prediction}</h2>
                <img src="/uploads/{file.filename}" width="300">
            </div>
            <div class="back-button">
                <a href="/" class="button">Back to Home Page</a>
            </div>            
            <footer>
                <p>&copy; 2025 DogCatClassifier - Kamil Haddad - All rights reserved</p>
            </footer>
        </body>
        </html>
        '''
    
    return '''
    <!doctype html>
    <html lang="sv">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dog or Cat?</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Classify">
        </form>
        <footer>
            <p>&copy; 2025 DogCatClassifier - All rights reserved</p>
        </footer>
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)
