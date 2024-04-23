from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530",token="root:Milvus")
client.create_collection(collection_name="faces", dimension=512, metric_type="COSINE")