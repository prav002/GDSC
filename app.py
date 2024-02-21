# app.py
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Replace 'known_face_encoding' with the encoding of a known person's face
# You would typically have a database of known faces and their encodings
known_face_encoding = np.array([])

def detect_hunger(image):
    # Find all face locations and face encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    is_hungry = True

    for face_encoding in face_encodings:
        # Compare the face encoding with the known face encoding
        results = face_recognition.compare_faces([known_face_encoding], face_encoding)
        
        if True in results:
            # If a known face is detected, the person is considered not hungry
            is_hungry = False
            break

    return is_hungry

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_hunger', methods=['POST'])
def process_image():
    try:
        image_data = request.files['image'].read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save the received image temporarily for debugging
        cv2.imwrite('received_image.jpg', image)
        
        is_hungry = detect_hunger(image)
        return jsonify({'is_hungry': is_hungry})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
