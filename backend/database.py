import motor.motor_asyncio
from datetime import datetime
from bson import ObjectId

client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client.summarization_db
summaries_collection = db.summaries

async def store_summary(original_text, summary, length, method):
    document = {
        "original_text": original_text,
        "summary": summary,
        "length": length,
        "method": method,
        "timestamp": datetime.utcnow()
    }
    await summaries_collection.insert_one(document)

async def get_summary_history():
    cursor = summaries_collection.find().sort("timestamp", -1).limit(10)
    history = await cursor.to_list(length=10)
    for item in history:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
    return history


# C:\Users\harsh\AppData\Local\Programs\mongosh\