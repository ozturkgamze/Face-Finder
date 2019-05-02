import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import pickle
from PIL import Image
import random

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	general_image = request.files['general_image']
	face_image = request.files['face_image']

	face_image_path = 'images/aranan-yuz/face_image.jpg'
	face_image.save(face_image_path)

	general_image_path = 'general_image.jpg'
	general_image.save(general_image_path)

	egit()

	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') #cascade oluşturulur.
	recognizer = cv2.face.LBPHFaceRecognizer_create() #yüz tanıma fonksiyonu recognizer değişkenine atanır.
	recognizer.read("trainner.yml") # eğitim bilgilerinin tutulduğu yml dosyası okunur.

	face_image_cv = cv2.imread(face_image_path) #klasördeki yüz fotoğrafını okuyup numpy_array türünde olarak face_image_cv değişkenine atar.
	general_image_cv = cv2.imread(general_image_path) #klasördeki genel fotoğrafı okuyup numpy_array türünde olarak general_image_cv değişkenine atar.

	frame = cv2.imread('general_image.jpg') # genel foto okunur
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #o fotoğraf griye dönüştürüldü
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # yüzler algılandı

	for(x, y, w, h) in faces:

		roi_gray = gray[y:y+h, x:x+h] #verilen koordinatlarda bulunan yüz roi_gray'e atanır
#		roi_color = frame[y:y+h, x:x+h] #roi_gray üzerinde tanımlama yapılır. Döndürülen oran ve id iki değişkene atanır.
		id_, oran = recognizer.predict(roi_gray)
		oran = int(oran) % 100
		if oran>=90 and oran <= 100:
			font = cv2.FONT_HERSHEY_DUPLEX
			
			name = '%'+str(oran)
			color = (0, 0, 255)
			stroke = 2
			cv2.putText(frame, name, (x + 40,y-5), font, 1, (0,225,0), stroke, cv2.LINE_AA)
			end_cord_x = x + w
			end_cord_y = y +h
			cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke) #verilen argümanlarla kırmızı kare çizdirildi.
		else:
			color = (255, 0, 0) #BGR
			stroke = 2 # line thickness
			
			name = '%'+str(oran)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (x + 40,y-5), font, 1, (0,255,0), stroke, cv2.LINE_AA)
			end_cord_x = x + w
			end_cord_y = y +h
			cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke) # verilen argümanlarla mavi kare çizdirildi.

	cv2.imwrite('results/result' + str(random.randint(0,100)) + '.jpg', frame)
	file_list = []

	for root,dirs, files in os.walk("results/"): # results dizinine yürür.
		for filename in files:
			file_list.append('results/' + filename)


	test_list = []
	for i in file_list:
		test = cv2.imread(i)
		list_content = cv2.imencode('.jpg', test)[1].tostring()
		encoded_list = base64.encodestring(list_content)
		to_send_list = 'data:image/jpg;base64, ' + str(encoded_list, 'utf-8')
		test_list.append(to_send_list) #sayfanın sağında yer alacak eski sonuçları gösterme işlemi.

	frame_content = cv2.imencode('.jpg', frame)[1].tostring() #frame'i jpg türünde encode'lar ve stringe dönüştürür.
	encoded_frame = base64.encodestring(frame_content) #base64 ü kullanarak aldığımız stringi encode'lar.
	to_send_frame = 'data:image/jpg;base64, ' + str(encoded_frame, 'utf-8')

	face_content = cv2.imencode('.jpg', face_image_cv)[1].tostring()  # face_image_cv'yi jpg türünde encode'lar ve stringe dönüştürür.
	encoded_face = base64.encodestring(face_content)  # base64 ü kullanarak aldığımız stringi encode'lar.
	to_send_face = 'data:image/jpg;base64, ' + str(encoded_face, 'utf-8')

	general_content = cv2.imencode('.jpg', general_image_cv)[1].tostring()  # general_image_cv'yi jpg türünde encode'lar ve stringe dönüştürür.
	encoded_general = base64.encodestring(general_content)  # base64 ü kullanarak aldığımız stringi encode'lar.
	to_send_general = 'data:image/jpg;base64, ' + str(encoded_general, 'utf-8')

	return render_template("index.html", init = True, result = to_send_frame, face = to_send_face, general = to_send_general, file=test_list) # fonk koştuysa init true olarak döner. html sayfasına sonuç fotosunu, genel fotoyu ve face fotosunu gönderir. + eski sonuçlar gider.

def egit():

	BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #app.py'nin olduğu dizini bulma işlemi
	image_dir = os.path.join(BASE_DIR, "images") #base_dir in içeriğine fotolar klasörünü dahil etme

	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create() #yüz tanıma fonksiyonu recognizer değişkenine atanır.

	current_id = 0
	label_ids = {}
	y_labels = []
	x_train = []

	for root, dirs, files in os.walk(image_dir): #roottaki(projedeki) fotoların da dahil olduğu image_dir e yürüyor. image_dir'in içinde root olur, dosyalar ve klasörler olabilir.
		for file in files: #dosyalar için dosya dosya geziyor.
			if file.endswith("png") or file.endswith("jpg"): #eğer dosyaların sonu png veya jpg ile bitiyorsa
				path = os.path.join(root, file) #o dosyanın yolunu al
				label = os.path.basename(root).replace(" ", "-").lower() #roottaki tüm klasörlerin adında boşluk varsa - koyup tüm harflerini küçük yapıyor. ve label değişkenine atıyor
				if not label in label_ids: #label id'lerinin içinde etiket yoksa
					label_ids[label] = current_id #label id'lerinde etiket oluşturur (current_id değeri ile)
					current_id +=1
				id_ = label_ids[label] # sonunda oluşan etiketler id_ değişkenine atandı.

				pil_image = Image.open(path).convert("L") #fotoğrafı griye dönüştürüp pil_image'e atıyor.
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS) #her fotoğrafı 550x550 boyutunda ölçeklendirip final_image'e atar.
				image_array = np.array(final_image, "uint8") #final_image'i 8 bitlik unsigned integer türüne dönüştürüp değerleri dizi haline getirip image_array değerine atar.

				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) #image_array içerisinde yüz algılama yapar.

				for(x, y, w, h) in faces:
					roi = image_array[y:y+h, x:x+w] #yüzün bulunduğu kareyi roi'ye atar.
					x_train.append(roi) #x_train dizisinin sonuna her seferinde roi'yi ekler.
					y_labels.append(id_) #y_labels dizisine de id_'ler eklenir

	with open("labels.pickle", 'wb') as f:
		pickle.dump(label_ids, f) #labels.pickle isminde dosya açıp içine etiket id'lerini ekler.

	recognizer.train(x_train, np.array(y_labels)) #x_train ve y_labels dizilerini merge edip eğitir.
	recognizer.save("trainner.yml") #eğitilenler trainner.yml dosyasına kaydedilir.

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(debug=True)
