from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'secret_key'

IMAGE_SIZE = 331

# Load trained model
sample_model = load_model('model5-01-1.06.keras')

labels = {
    0: 'Hyundai', 1: 'Lexus', 2: 'Mazda', 3: 'Mercedes',
    4: 'Opel', 5: 'Skoda', 6: 'Toyota', 7: 'Volkswagen'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'logo_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['logo_image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.expand_dims(resized, axis=0)
            result = sample_model.predict(img_array)
            predicted_label = labels[np.argmax(result)]
            flash(f'Predicted Logo: {predicted_label}')
            return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)