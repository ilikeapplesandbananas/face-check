"""Seed script: Download and index verified public faces for immediate testing."""

import sys
import httpx
from app.db.session import create_session_factory, initialize_schema
from app.services.face_pipeline import FacePipeline
from app.services.vector_index import FaceIndexer

SAMPLES = [
    {
        "name": "Barack Obama",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/440px-President_Barack_Obama.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Barack_Obama"
    },
    {
        "name": "Elon Musk",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Elon_Musk_Colorado_2022_%28cropped2%29.jpg/440px-Elon_Musk_Colorado_2022_%28cropped2%29.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Elon_Musk"
    },
    {
        "name": "Taylor Swift",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/191125_Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.png/440px-191125_Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.png",
        "page_url": "https://en.wikipedia.org/wiki/Taylor_Swift"
    },
    {
        "name": "Keanu Reeves",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Reuni%C3%A3o_com_o_ator_norte-americano_Keanu_Reeves_%28cropped%29.jpg/440px-Reuni%C3%A3o_com_o_ator_norte-americano_Keanu_Reeves_%28cropped%29.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Keanu_Reeves"
    },
    {
        "name": "Satya Nadella",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/MS-Exec-Nadella-Satya-2017-08-31-22_%28cropped%29.jpg/440px-MS-Exec-Nadella-Satya-2017-08-31-22_%28cropped%29.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Satya_Nadella"
    },
    {
        "name": "Sundar Pichai",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Sundar_Pichai_crop.jpg/440px-Sundar_Pichai_crop.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Sundar_Pichai"
    },
    {
        "name": "Sam Altman",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Sam_Altman_TechCrunch_Disrupt_2019_%28cropped%29.jpg/440px-Sam_Altman_TechCrunch_Disrupt_2019_%28cropped%29.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Sam_Altman"
    },
    {
        "name": "Jensen Huang",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Jensen_Huang_at_Computex_2024.jpg/440px-Jensen_Huang_at_Computex_2024.jpg",
        "page_url": "https://en.wikipedia.org/wiki/Jensen_Huang"
    }
]

def seed():
    print("Initializing schema...")
    initialize_schema()
    pipeline = FacePipeline()
    session_factory = create_session_factory()
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    indexed_count = 0

    with session_factory() as session:
        indexer = FaceIndexer(session)
        with httpx.Client(headers=headers, follow_redirects=True, timeout=15) as client:
            for item in SAMPLES:
                print(f"Downloading & indexing {item['name']}...")
                try:
                    r = client.get(item["image_url"])
                    if r.status_code != 200:
                        print(f"  Failed to fetch: {r.status_code}")
                        continue
                    
                    emb = pipeline.extract(r.content)
                    rec = indexer.index(
                        embedding=emb.vector,
                        source_image_url=item["image_url"],
                        page_context_url=item["page_url"]
                    )
                    print(f"  SUCCESS: Profile {rec.profile_id} indexed!")
                    indexed_count += 1
                except Exception as exc:
                    print(f"  Error: {exc}")
                    
    print(f"\nDone! Indexed {indexed_count} public faces into Qdrant memory.")

if __name__ == "__main__":
    seed()
