from flask import Flask,render_template,request,jsonify,session
import numpy as np, cv2
from tensorflow import keras
from util_ import display_mask, get_boundry_img_matrix
from PCA import pca
import pickle
from segment_formation_v6 import segment_image4
import classifier_2_v2,Number_grain_detect
from sklearn import metrics
from sklearn.metrics import accuracy_score
import os
folder = os.path.join('static')
app = Flask(__name__)

app.config['UPLOAD_FOLDER']=folder
app.secret_key="abc"
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/estimate',methods = ['POST'])
def estimate():
    if request.method == 'POST':
        url = os.path.join(app.config['UPLOAD_FOLDER'],request.form['file'])
        phno = request.form['phno']
        session['phno'] = phno   
    np.warnings.filterwarnings('ignore')
    color = {i: np.random.randint(20, 255, 3) for i in range(5, 5000)}
    color[1] = [255, 255, 255]
    color[2] = [0, 0, 255]
    imgFile =  url
    # imgFile = 'test_2.jpg'
    count = 1

    model = keras.models.load_model('weights_results_2out/weights_01234567.h5')
    np.warnings.filterwarnings('ignore')
    # for imgFile in imgFile:
    print("Segmentation in process...")
    np.warnings.filterwarnings('ignore')
    segments, segLocation, _, mask= segment_image4(imgFile)
    np.warnings.filterwarnings('ignore')
    print("Segmentation in Complete.")
    features = {}
    print("\n\nFeature extraction in process...")
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    k=0
    for gi in segments:
        gcolor = segments[gi]
        h, w, _ = gcolor.shape
        ggray = gcolor[:,:,2]
        thresh = np.array([[255 if pixel > 20 else 0 for pixel in row] for row in ggray])
        b = np.array(get_boundry_img_matrix(thresh, bval=1), dtype=np.float32)
        boundry = np.sum(b) / (h * w)
        area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i, j]] for i in range(h)]))
        mean_area = area / (h * w)
        r, b, g = np.sum([gcolor[i, j] for j in range(w) for i in range(h)], axis=0) / (area * 256)
        _, _, eigen_value = pca(ggray)
        eccentricity = eigen_value[0] / eigen_value[1]
        l = [mean_area, boundry, r, b, g, eigen_value[0], eigen_value[1], eccentricity]
        features[gi] = np.array(l)
        k+=1
    print("Feature extraction in complete.")
    ftrain, y_train,ftest, y_test = pickle.load(open("weights_results_2out/grain_feature.pkl", 'rb'), encoding="bytes")
    model.fit(np.array(ftrain),np.array(y_train))
    out = {}
    for i in features:
        out[i] = model.predict(np.array([features[i]]))
    y_actual= []
    for i in range(10):
        y_actual.append(np.argmax(y_test[i]))
    y_pred = []
    rect = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    good = not_good = 0
    count = 0
    for i in out:
        try:
            s = segLocation[i]
        except KeyError:
            print("Key Error")
            continue
        if np.argmax(out[i][0]) == 0:
            good += 1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 0), 1)
        else:
            not_good+=1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 255), 3)
        if count<10 :
            y_pred.append(np.argmax(out[i][0]))
        count+=1
    cm = metrics.confusion_matrix(y_actual,y_pred)
    cm_nom = cm.astype('float')/cm.sum(axis=1)
    print(cm_nom)
    print(metrics.classification_report(y_actual,y_pred))
    print("\n\nNumber of good grain :", good)
    print("Number Not good grain or imputity:", not_good)

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = rect.shape
    cv2.putText(rect, text='Number of good grain: %d  Number Not good grain or imputity: %d'%(good,not_good), org=(10, h - 50), fontScale=1, fontFace=font, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


    
    maskFile = 'mask_'+imgFile.split('/')[-1]
    outFile = 'result_'+imgFile.split('/')[-1]
    cv2.imwrite(outFile, rect)
    # display_mask('mask',mask,sname=maskFile)
    # cv2.waitKey(0)
    count+=1
    return render_template("display.html",url=url,pure=good,impure=not_good)
@app.route('/mlp',methods=['GET'])
def mlp():
    score = classifier_2_v2.mlp()
    return jsonify(result=score)
@app.route('/cnn',methods=['GET'])
def cnn():
    score = Number_grain_detect.cnn()
    return jsonify(result=score)
@app.route('/nog',methods=['GET'])
def nog():
    oneGrain = 'segmentation_data/boundry_1_30.pkl'
    moreGrain = 'segmentation_data/boundry_2_30.pkl'
    oneGrain_list = pickle.load(open(oneGrain,'rb'), encoding="latin")
    moreGrain_list = pickle.load(open(moreGrain,'rb'), encoding="latin")
    Grain = oneGrain_list + moreGrain_list
    print("Number of one grain sample:", len(oneGrain_list))
    print("Number of more grain sample:",len(moreGrain_list))
    print("Total of sample:",len(Grain))
    score = [len(oneGrain_list),len(moreGrain_list),len(Grain)]
    return jsonify(result=score)
@app.route('/logout',methods=['GET'])
def logout():
    if 'phno' in session:
        session.pop('phno',None)
        return render_template("index.html")
    else:
        return '<p>Login first</p>'
if __name__ == "__main__":
    app.run()
