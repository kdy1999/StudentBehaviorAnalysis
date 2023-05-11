from socket import *
import time
import os,base64
import base64
import numpy as np
import cv2
from flask import Flask, request ,Response
import random
import string
import json

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    if request.method == 'POST':
        base64_data = request.json['images']
        imgdata=base64.b64decode(str(base64_data))
        file=open('/home/ai/StudentBehaviorAnalysis-v0.1.6/img/1.jpg','wb')
        file.write(imgdata)
        file.close()
        return socket_send_thread()
    return '400'

def socket_send_thread():
    ip_port = (('localhost', 8037))
    while True:
        try:
            serversocket = socket(AF_INET, SOCK_STREAM)
            print("build socket")
            break
        except ConnectionError as e:
            continue
    while True:
        try:
            serversocket.connect(ip_port)
            print("successed to connect")
            break
        except ConnectionError as e:
            print("failed to connect")
            continue

    serversocket.send("/home/ai/StudentBehaviorAnalysis-v0.1.6/img/1.jpg".encode("utf-8"))
    serversocket.send("inter".encode("utf-8"))
    time.sleep(1)
    data_recieved = ''
    while "inter" not in data_recieved:
        data, addr = serversocket.recvfrom(10000)
        if not data:
            print("connet interrupt")
            break
        data_recieved += str(data, encoding="utf-8")
    data_tmp = data_recieved.split("inter")
    data_recieved = data_tmp[-1]
    res = json.loads(data_tmp[0])
    return res["results"][0]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
