"""Index a vetted local face image and its source metadata.

Example:
python -m scripts.index_face --image-file /data/face.jpg \
  --source-image-url https://example.org/image.jpg \
  --page-context-url https://example.org/article
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID

from app.db.session import create_session_factory, initialize_schema
from app.services.face_pipeline import FacePipeline
from app.services.vector_index import FaceIndexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a face embedding with its URL metadata.")
    parser.add_argument("--image-file", type=Path, required=True, help="Local image to process through the strict pipeline.")
    parser.add_argument("--source-image-url", required=True, help="Direct URL of the indexed source image.")
    parser.add_argument("--page-context-url", required=True, help="Page URL where the source image appeared.")
    parser.add_argument("--profile-id", type=UUID, help="Existing profile UUID to append a verified sample to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_file.is_file():
        raise SystemExit(f"Image file not found: {args.image_file}")

    embedding = FacePipeline().extract(args.image_file.read_bytes())
    initialize_schema()
    session_factory = create_session_factory()
    with session_factory() as session:
        record = FaceIndexer(session).index(
            embedding=embedding.vector,
            source_image_url=args.source_image_url,
            page_context_url=args.page_context_url,
            profile_id=args.profile_id,
        )
    print(f"Indexed profile={record.profile_id} face={record.indexed_face_id} vector={record.vector_id}")


if __name__ == "__main__":
    main()
