import os
import uuid
from flask import Flask, flash, request, redirect , render_template
import librosa
import librosa.display
import numpy as np
from keras.models import load_model
import keras
import numpy as np
from keras.preprocessing import image
keras.backend.clear_session()


UPLOAD_FOLDER = 'files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# class names with indices
class_labels = {  0:'Applause', 1:'Bus', 2:'Computer_keyboard', 3:'Cough', 4:'Fireworks',
                  5:'Keys_jangling', 6:'Laughter', 7:'Microwave_oven', 8:'Scissors',
                  9:'Tearing', 10:'Telephone', 11:'air_conditioner', 12:'car_horn',
                  13:'children_playing', 14:'dog_bark', 15:'drilling', 16:'engine_idling',
                  17:'gun_shot', 18:'jackhammer', 19:'siren', 20:'street_music'}

# load the trained model from saved weights
sound_classify = load_model('model_weights/sound_classification_model.h5')


def predict_label( mfcc_features ):
    try:

        batch_size = 1
        sound_feature = np.concatenate([ mfcc_features[np.newaxis, : ]] * batch_size)

		# model endpoint inference
        model_response = sound_classify.predict( sound_feature )

        i_idx = np.argmax( model_response[0])
        print(class_labels[ i_idx ])
        return class_labels[ i_idx ]
    except:
        return None


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".mp3"
    global full_file_name 
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    print( full_file_name )
    file.save(full_file_name)
    return '<h1>Success</h1>'

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    try:
        if request.method == 'POST':

            x, sr = librosa.load( full_file_name )
            mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=40)
            mfcc_scaled = np.mean(mfccs.T,axis=0)

            p = predict_label(  np.array(mfcc_scaled) )

        return render_template("predict.html", prediction = p )
    except:
        return render_template("predict.html" , prediction = "Error In Preidction ... Please Retry !!"  )

if __name__ == '__main__':
    app.run()