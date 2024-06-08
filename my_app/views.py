from urllib import request as urlrequest
from django.shortcuts import render, redirect
from binascii import a2b_base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from django.http import HttpResponse

from .models import Song



def open_camera(request):
    return render(request, 'camera.html')

def get_label(argument):
    labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return labels.get(argument, "Invalid emotion")


def get_songs(mood):
    all_songs = Song.objects.all()
    mood_songs = []
    for song in all_songs:
        if song.mood == mood:
            mood_songs.append(song)
    return mood_songs


def play_song_from_cloudinary(song_url):
    try:
        # Assuming song_url is the direct Cloudinary URL of the song
        response = urlrequest.urlopen(song_url)
        song_content = response.read()

        # Prepare the HTTP response with the song content
        response = HttpResponse(song_content, content_type='audio/mpeg')
        response['Content-Disposition'] = 'inline'
        return response

    except Exception as e:
        return HttpResponse(f'Error: Unable to retrieve song from Cloudinary - {str(e)}', status=404)


def detect(request):
    my_model = load_model("my_app/my_model.hdf5", compile=False)
    face_cascade = cv2.CascadeClassifier('my_app/haarcascade_frontalface_default.xml')

    try:
        data = request.POST['image_data']
    except KeyError:
        K.clear_session()
        return redirect(open_camera)

    binary_data = a2b_base64(data)
    image_data = BytesIO(binary_data)
    img = np.array(Image.open(image_data))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        K.clear_session()
        return redirect(open_camera)

    for (x, y, w, h) in faces:
        crop_img = img[y:y + h, x:x + w]
        break  # Use the first detected face only

    test_image = cv2.resize(crop_img, (64, 64))
    test_image = np.array(test_image)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    gray = gray / 255
    gray = gray.reshape(-1, 64, 64, 1)

    res = my_model.predict(gray)
    result_num = np.argmax(res)
    mood = get_label(result_num)

    K.clear_session()

    context = {'my_mood': mood, 'mood_songs': get_songs(mood)}

    if 'song_url' in request.POST:
        song_url = request.POST['song_url']
        if song_url:
            # Call the function to play the song using the complete URL
            return play_song_from_cloudinary(song_url)
        else:
            return HttpResponse('Error: Song URL not provided.', status=400)

    return render(request, 'main2.html', context)


