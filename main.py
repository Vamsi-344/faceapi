import base64
import os
import sys
from fastapi import FastAPI, File, UploadFile, Depends, Form

import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.mtcnn import MTCNN
from src.inception_resnet_v1 import InceptionResnetV1
from src.data_io import transform as trans
from src.generate_patches import CropImage
from test import load_pretrained_fas_model, crop_image_with_ratio

from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
import uvicorn
import time

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="FaceAPI", description="Deploying a FaceAPI with the combination of MTCNN(for face detection), FaceNet(for face Recognition) and MiniFAS(for face anti spoofing) models for face recognition system")

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
error_logger = logging.getLogger("uvicorn:error")
# error_logger.setLevel(logging.INFO)
error_logger.propagate = False
# stream_handler = logging.StreamHandler(sys.stdout)
# log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
# stream_handler.setFormatter(log_formatter)
# logger.addHandler(stream_handler)

def check_spoof(img, img_aligned, boxes):
    boxes_int=boxes.astype(int)
    spoofs = []
    if img_aligned is not None:
        for idx, (left,top, right, bottom) in enumerate(boxes_int):
            img=crop_image_with_ratio(img,4,3,(left+right)//2)
            image_cropper = CropImage()
            image = cv2.resize(img, (int(img.shape[0] * 3 / 4), img.shape[0]))
            boxes, confs, landmarks = mtcnn.detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), landmarks=True)
            image_bbox = [int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]-boxes[0][0]+1), int(boxes[0][3]-boxes[0][1]+1)]
            prediction = np.zeros((1, 3))
            for minifas_model, minifas_model_params in zip(minifas_models, minifas_models_params):
                minifas_model_params["org_img"] = image
                minifas_model_params["bbox"] = image_bbox
                img = image_cropper.crop(**minifas_model_params)
                test_transform = trans.Compose([
                    trans.ToTensor(),
                ])
                img = test_transform(img)
                img = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    result = minifas_model.forward(img)
                    result = F.softmax(result).cpu().numpy()
                    prediction += result
            label = np.argmax(prediction)
            value = prediction[0][label]/2
            if label==1:
                spoofs.append("REAL")
            else:
                spoofs.append("FAKE")
    if 'FAKE' in spoofs:
        return 0
    else:
        return 1

@app.on_event("startup")
async def on_startup():
    try:
        global mtcnn
        global resnet
        global minifas_models, minifas_models_params
        global device
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        mtcnn = MTCNN()
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        minifas_v2, minifas_v2_params = load_pretrained_fas_model(model_path='pretrained_models/2.7_80x80_MiniFASNetV2.pth', device=device)
        minifas_v1_se, minifas_v1_se_params = load_pretrained_fas_model(model_path='pretrained_models/4_0_0_80x80_MiniFASNetV1SE.pth', device=device)
        minifas_v1_se.eval()
        minifas_v2.eval()
        minifas_models = [minifas_v1_se, minifas_v2]
        minifas_models_params = [minifas_v1_se_params, minifas_v2_params]
        logger.info(f'Successfully loaded the pretrained models')
    except Exception as e:
        error_logger.debug(f'Failed to load models with {e}')
    try:
        time.sleep(int(os.environ['SLEEP_FOR']))
        global client
        client = MilvusClient(uri=os.environ['MILVUS_URL'])
        logger.info("Created a milvus client successfully")
    except Exception as e:
        error_logger.debug(f'Failed to create a Milvus Client with {e}')

    try:
        logger.info("Trying to load collection named faces")
        client.load_collection(collection_name="faces")
    except Exception as e:
        error_logger.debug(f'Failed to load collection named faces with {e}')
    
    try:
        res = client.has_collection(collection_name="faces")
        if res:
            logger.info(f'Has the faces collection')
        else:
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )

            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)

            # 3.3. Prepare index parameters
            index_params = client.prepare_index_params()

            # 3.4. Add indexes
            # index_params.add_index(
            #     field_name="id",
            #     index_type="STL_SORT"
            # )

            index_params.add_index(
                field_name="vector", 
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={ "nlist": 128 }
            )
            client.create_collection(collection_name="faces", schema=schema, description="Similar faces search", index_params=index_params)
            client.load_collection(collection_name="faces")
            # face_id = FieldSchema(
            # name="id", 
            # dtype=DataType.VARCHAR,
            # max_length=128, 
            # is_primary=True, 
            # )
            # face_info = FieldSchema(
            # name="vector", 
            # dtype=DataType.FLOAT_VECTOR, 
            # dim=512
            # )
            # schema = CollectionSchema(
            # fields=[face_id, face_info], 
            # description="Similar face search"
            # )
            # # collection_name = "faces"
            # client.create_collection(collection_name="faces", schema=schema, metric_type="COSINE", enable_dynamic_field=True, description="Similar face search", index)
            # client.load_collection(collection_name="faces")
            # logger.info("Successfully created & loaded faces collection")
            # client._create_collection_with_schema(collection_name=collection_name, schema=schema, metric_type="COSINE", enable_dynamic_field=True)
            # client.create_collection(collection_name="faces", dimension=512, metric_type="COSINE", auto_id=True, enable_dynamic_field=True)
    except Exception as e:
        error_logger.debug(f'Failed to create faces collection with {e}')
    
@app.on_event("shutdown")
async def on_shutdown():
    try:
        del mtcnn, resnet
    except Exception as e:
        logger.info(f'Failed to unload models with {e}')
    
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
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the image data to a PIL Image
        # img = Image.open(io.BytesIO(image_data))
    else:
        contents = await body.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_aligned, conf = mtcnn(img_rgb, return_prob=True)

    if img_aligned is None:
        return {"facesDetected": 0, "response": "unable to detect the face"}

    processed["image"] = img_aligned.tolist()
    processed["confidence"] = conf

    boxes, confs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    n_faces = len(boxes)
    spf = check_spoof(img=img, img_aligned=img_aligned, boxes=boxes)
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

    return {"spoofResult": spoofResult, "facesDetected": n_faces, "detection": detection, "largestfaceInfo": processed}

@app.post("/feature_extract")
async def do_feature_extract(img_info: dict = Depends(do_detect)):
    if img_info["facesDetected"] == 0:
        return {"response": "no faces are detected, please try again with better picture"}
    elif img_info["spoofResult"] == "fake":
        return {"response": "antispoofing attack detected, please try again without any attack"}
    else:
        # if img_info["facesDetected"]>1:
        #     return {"response": "more than one face is detected,please try again"}
        # else:
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
async def do_register(img_info: dict = Depends(do_feature_extract), username: str = None, realm: str = None):
    # employee_id: int = None
    if type(img_info["response"])==str:
        return {"response": "Can't perform recognition due to feature extraction result: "+ img_info["response"]}
    else:
        # if employee_id is None:
        #     return {"response": "Employee Id shouldn't be None"}
        if username is None:
            return {"response": "username shouldn't be None"}
        elif realm is None:
            return {"response": "realm shouldn't be None"}
        else:
            try:
                id = username+"_"+realm
                # data = [{"id": employee_id, "vector": img_info["response"], "name": name}]
                data = [{"id": id, "vector": img_info["response"], "username": username, "realm": realm}]
                res = client.upsert(collection_name="faces", data=data)
                return {"message": "Face registered successfully"}
            except Exception as e:
                return {"message": "Error registering face", "error": str(e)}
            # return img_info

@app.get("/get_data")
async def get_data():
    res = client.query(collection_name="faces", filter="id like '%'", output_fields=["username", "realm"])
    return {"results": res}

@app.get("/get_face/")
async def get_face(username: str):
    res = client.query(collection_name="faces", filter=f"id like '{username}_%'", output_fields=["username", "realm"])
    # res = client.get(collection_name="faces", ids=[id], output_fields=["username"])
    return {"response": "Face found on database", "result": res}

@app.delete("/delete_face/")
async def delete_face(id: str):
    res = client.delete(collection_name="faces", ids=[id])
    return {"response": "Face deleted from database", "id": id}

if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
