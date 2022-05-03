import os
import cv2
from os import listdir
from os.path import isfile, join
classifier = load_model('vgg16f.h5')
five_celeb_dict = {"[0]": "Akhil", 
                      "[1]": "Apurva",
                      "[2]": "Jai",
                      "[3]": "Mahir",
                      "[4]": "Manan",
                      "[5]": "Mody",
                      "[6]": "Omkar",
                      "[7]": "Rahi",
                      "[8]": "Sidharth",
                      "[9]": "Sneha",
                      "[10]": "Soham"
                      
                     }

five_celeb_dict_n = { "Akhil": "Akhil", 
                      "Apurva": "Apurva",
                      "Jai": "Jai",
                      "Mahir": "Mahir",
                      "Manan": "Manan",
                      "Mody": "Mody",
                      "Omkar": "Omkar",
                      "Rahi": "Rahi",
                      "Sidharth": "Sidharth",
                      "Sneha": "Sneha",
                      "Soham": "Soham"
                      
                      }

def draw_test(name, pred, im):
    celeb =five_celeb_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, celeb, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + five_celeb_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("Datasets/Test/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    print(classifier.predict(input_im))
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

    
cv2.destroyAllWindows()