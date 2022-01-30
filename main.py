from flask import Flask, render_template, Response, request, jsonify
import cv2
#Librerias de analisis
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

app=Flask(__name__)
Tabla=[]
"Iniciamos CAMARA"
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml")

def generate():
    while(True):
        "Capturamos Video Frame a Frame"
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y),(x + w, y+h),(0, 255, 0), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')

def DetectFaces(image_file):
    print('Detectando rostros en', image_file)

    # Specify facial features to be retrieved
    features = [FaceAttributeType.age,
                FaceAttributeType.gender,
                FaceAttributeType.emotion,
                FaceAttributeType.glasses]

    # Get faces
    with open(image_file, mode="rb") as image_data:
        global Tabla
        detected_faces = face_client.face.detect_with_stream(image=image_data,
    return_face_attributes=features)

        if len(detected_faces) > 0:
            Tabla.append(str(len(detected_faces))+' rostros detectados')

            # Prepare image for drawing
            fig = plt.figure(figsize=(8, 6))
            plt.axis('off')
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            color = 'lightgreen'

            # Draw and annotate each face
            for face in detected_faces:

                # Get face properties
                Tabla.append('Rostro ID: {}'.format(face.face_id))
                detected_attributes = face.face_attributes.as_dict()
                age = 'age unknown' if 'age' not in detected_attributes.keys() else int(detected_attributes['age'])
                Tabla.append(' - Edad: {}'.format(age))
                
                if 'gender' in detected_attributes:
                    if format(detected_attributes['gender'])=="male":
                        Tabla.append('-Sexo: Masculino')
                    else:
                        Tabla.append('-Sexo: Femenino')

                if 'emotion' in detected_attributes:
                    Tabla.append('--- Emociones ---')
                    Español=['-Enfado','-Desprecio','-Asco','-Miedo', '-Felicidad','-Neutral','-Tristeza','-Sorpresa']
                    i=0
                    for emotion_name in detected_attributes['emotion']:
                        Tabla.append('   - {}: {}'.format(Español[i], detected_attributes['emotion'][emotion_name]))
                        i+=1
                        
                if 'glasses' in detected_attributes:
                    Tabla.append(' - Anteojos: {}'.format(detected_attributes['glasses']))

                # Draw and annotate face
                r = face.face_rectangle
                bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
                draw = ImageDraw.Draw(image)
                draw.rectangle(bounding_box, outline=color, width=5)
                annotation = 'Face ID: {}'.format(face.face_id)
                plt.annotate(annotation,(r.left, r.top), backgroundcolor=color)

            # Save annotated image
            plt.imshow(image)
            outputfile = 'detected_faces.jpg'
            fig.savefig(os.path.join('./static/images', outputfile))

            print('\nResults saved in', outputfile)
            return Tabla


@app.route('/')
def index():
    #return "!HOLA mundo¡"
    #return render_template('index.html')
    data={
        'titulo':'Index',
        'bienvenida':'!Saludos¡'
    }
    return render_template('index.html',data=data)

@app.route('/Captura')
def Captura():
    return render_template('Captura.html')

@app.route("/tomar_foto_guardar")
def guardar_foto():
    nombre_foto ="Rostro.jpg"
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(os.path.join('./static/images/Rostro.jpg'), frame)
    return jsonify({
        "ok": ok,
        "nombre_foto": nombre_foto,
    })

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/analisis")
def Analisis():
    global face_client
    global Tabla
    Tabla=[]
    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
        cog_key = os.getenv('COG_SERVICE_KEY')

        # Authenticate Face client
        credentials = CognitiveServicesCredentials(cog_key)
        face_client = FaceClient(cog_endpoint, credentials)

        Tabla=DetectFaces(os.path.join('./static/images/Rostro.jpg'))              
    except Exception as ex:
        print(ex)
    Datos= {
            'Emociones':Tabla,
            'Num_Emociones':len(Tabla)
            }
    return render_template('analisis.html',Datos=Datos)


@app.route('/contacto/<nombre>')
def contacto(nombre):
    data={
        'titulo':'Contacto',
        'nombre':nombre
    }
    return render_template('contacto.html', data=data)

def pagina_no_encontrada(error):
    return render_template('404.html'), 404
    #return redirect(url_for('index'))

if __name__== '__main__':
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug=True, port=5000)
Tabla=[]
cap.release()
