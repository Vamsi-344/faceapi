import base64
import io
import os
from fastapi import FastAPI, File, UploadFile, Depends, Form

from PIL import Image

import numpy as np
import torch

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from test import check_spoof

from pymilvus import MilvusClient
import uvicorn
import time

app = FastAPI(title="FaceAPI", description="Deploying a FaceAPI with the combination of MTCNN(for face detection), FaceNet(for face Recognition) and MiniFAS(for face anti spoofing) models for face recognition system")

@app.on_event("startup")
async def on_startup():
    try:
        global mtcnn
        global resnet
        mtcnn = MTCNN()
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
    except Exception as e:
        print(f'Failed to load models with {e}')
    try:
        time.sleep(int(os.environ['SLEEP_FOR']))
        global client
        client = MilvusClient(uri=os.environ['MILVUS_URL'])
        client.load_collection(collection_name="faces")
    except Exception as e:
        print(f'Failed to create a Milvus Client with {e}')
    
    try:
        res = client.has_collection(collection_name="faces")
        if res:
            print(f'Has the faces Collection')
        else:
            client.create_collection(collection_name="faces", dimension=512, metric_type="COSINE", auto_id=True, enable_dynamic_field=True)
    except Exception as e:
        print(f'Failed to create collection with {e}')
    
@app.on_event("shutdown")
async def on_shutdown():
    try:
        del mtcnn, resnet
    except Exception as e:
        print(f'Failed to unload models with {e}')
    
@app.get("/")
async def root():
    return {"Welcome to FaceAPI"}

@app.post("/detect")
async def do_detect(body: UploadFile=File(None), base64_string: str = Form(None)):
    processed = {}
    detection = {}

    if base64_string:
        # Read the image data
        image_data = base64.b64decode(base64_string)
        # image_data = await base64_string.read()
        
        # Convert the image data to a PIL Image
        img = Image.open(io.BytesIO(image_data))
    else:
        img = Image.open(body.file)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
    # Convert the image to RGB mode, which removes the alpha channel
        img = img.convert('RGB')
    img = np.array(img)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    # print(img.shape)
    
    img_aligned, conf = mtcnn(img, return_prob=True)

    if img_aligned is None:
        return {"facesDetected": 0, "response": "unable to detect the face"}

    processed["image"] = img_aligned.tolist()
    processed["confidence"] = conf

    boxes, confs, landmarks = mtcnn.detect(img, landmarks=True)
    n_faces = len(boxes)
    spf = check_spoof(image_array=img, face_locations=img_aligned, boxes=boxes)
    values = []
    for box, conf, landmark in zip(boxes, confs, landmarks):
        face_instance = {}
        face_instance["boundingBox"] = {}
        face_instance["boundingBox"]["x0"] = box[0]
        face_instance["boundingBox"]["y0"] = box[1]
        face_instance["boundingBox"]["x1"] = box[2]
        face_instance["boundingBox"]["y1"] = box[3]

        face_instance["confidence"] = conf

        face_instance["landmarks"] = {}
        face_instance["landmarks"]["leftEye"] = landmark[0].tolist()
        face_instance["landmarks"]["rightEye"] = landmark[1].tolist()
        face_instance["landmarks"]["nose"] = landmark[2].tolist()
        face_instance["landmarks"]["mouthleftCorner"] = landmark[3].tolist()
        face_instance["landmarks"]["mouthrightCorner"] = landmark[4].tolist()
        
        values.append(face_instance)
    detection["values"] = values

    # embedding = resnet(torch.stack([img_aligned]))[0]

    if spf==0:
        spoofResult = "fake"
    else:
        spoofResult = "real"

    return {"spoofResult": spoofResult,"facesDetected": n_faces, "detection": detection, "largestfaceInfo": processed}

@app.post("/feature_extract")
async def do_feature_extract(img_info: dict = Depends(do_detect)):
    if img_info["facesDetected"] == 0:
        return {"response": "no faces are detected, please try again with better picture"}
    elif img_info["spoofResult"] == "fake":
        return {"response": "antispoofing attack detected, please try again without any attack"}
    else:
        if img_info["facesDetected"]>1:
            return {"response": "more than one face is detected,please try again"}
        else:
            img_aligned = torch.from_numpy(np.array(img_info["largestfaceInfo"]["image"], dtype=np.float32))
            conf = np.array(img_info["largestfaceInfo"]["confidence"])
            face_embedding = resnet(torch.stack([img_aligned]))[0]
            return {"response": face_embedding.tolist()}

@app.post("/recognize")
async def do_recognize(img_info: dict = Depends(do_feature_extract)):
    if type(img_info["response"])==str:
        return {"response": "Can't perform registration due to feature extraction result: "+ img_info["response"]}
    else:
        search_params = {"metric_type": "COSINE", "params": {"radius": 0.7, "range_filter": 1.1}}
        res = client.search(collection_name="faces", data=[img_info["response"]], limit=1, output_fields=["name"], search_params=search_params)
        if res[0]==[]:
            return {"message": "Face not recognized", "results": "Unknown"} 
        return {"message": "Face recognized", "results": res}
    
@app.post("/register")
async def do_register(img_info: dict = Depends(do_feature_extract), name: str = None):
    # employee_id: int = None
    if type(img_info["response"])==str:
        return {"response": "Can't perform recognition due to feature extraction result: "+ img_info["response"]}
    else:
        # if employee_id is None:
        #     return {"response": "Employee Id shouldn't be None"}
        if name is None:
            return {"response": "Name shouldn't be None"}
        else:
            try:
                # data = [{"id": employee_id, "vector": img_info["response"], "name": name}]
                data = [{"vector": img_info["response"], "name": name}]
                res = client.insert(collection_name="faces", data=data)
                return {"message": "Face registered successfully"}
            except Exception as e:
                return {"message": "Error registering face", "error": str(e)}
            # return img_info

@app.get("/get_data")
async def get_data():
    res = client.query(collection_name="faces", filter="id>0", output_fields=["name"])
    return {"results": res}

@app.get("/get_face/")
async def get_face(employee_id: int):
    res = client.get(collection_name="faces", ids=[employee_id], output_fields=["name"])
    return {"response": "Face found on database", "result": res}

@app.delete("/delete_face/")
async def delete_face(employee_id: int):
    res = client.delete(collection_name="faces", ids=[employee_id])
    return {"response": "Face deleted from database", "employee_id": employee_id}

if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)