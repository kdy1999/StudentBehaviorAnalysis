##base64转cv2
import base64
import numpy as np
import cv2

def cv2_base64(image):
    base64_str = cv2.imencode('6.jpg',image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    print(base64_str)
    return base64_str

def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString,np.uint8)
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    cv2.imshow('image',image)
    return image

if __name__ == "__main__":
    image = cv2_base64(cv2.imread('6.jpg'))
    print(image)
    cv2.imshow('image',base64_cv2(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()