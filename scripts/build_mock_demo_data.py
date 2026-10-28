"""Generate a lightweight offline mock catalog for local demo when the full 10GB dataset is absent.

Writes into tmp/mock_demo_data by default (ignored by Git, never overwrites data/processed/).
Run: python scripts/build_mock_demo_data.py
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'tmp/mock_demo_data'

MOCK_ITEMS = [
    # Warm items (train_count > 20)
    ('B006GBITXC', 185, 'LEGO Classic Medium Creative Brick Box 10696 Building Toy Set', 'LEGO classic creative bricks building toy construction blocks kids imagination'),
    ('B07PXGQC1Q', 142, 'LEGO Star Wars Millennium Falcon 75257 Building Kit Starship', 'LEGO star wars millennium falcon spaceship building blocks construction toy'),
    ('B08HW1L75J', 118, 'Ravensburger Disney Castle 1000 Piece Jigsaw Puzzle for Adults & Kids', 'Ravensburger jigsaw puzzle 1000 piece board family brain teaser castle'),
    ('B07W7XYZ12', 96, 'Melissa & Doug Wooden Building Blocks Set 100 Pieces Montessori Educational', 'Montessori wooden blocks educational preschool learning fine motor skills'),
    ('B084Z3X99K', 88, 'Squishmallows Official Kellytoy Plush 16 Inch Soft Stuffed Animal Pillow Toy', 'Plush stuffed animal soft pillow cuddle kids cute kawaii toy'),
    ('B01N4Q5X8M', 74, 'Catan The Board Game Strategy Settlers Family Tabletop Classic', 'Board game strategy tabletop family card resource trading'),
    ('B07V5G8LPQ', 65, 'LEGO Technic McLaren Senna GTR 42123 Racing Car Model Kit', 'LEGO technic racing car engineering blocks model building toy'),
    ('B09B2W7R4T', 53, 'GUND Baby Animated Flappy The Elephant Plush Singing Stuffed Toy', 'Plush elephant baby stuffed animal soft singing cuddle toy'),
    ('B08K3M9P1V', 44, 'Melissa & Doug Wooden Jigsaw Puzzles in a Box - Dinosaurs 4-Pack', 'Wooden jigsaw puzzle dinosaur educational kids preschool brain teaser'),
    ('B07Y8Q2N5L', 31, 'Montessori Wooden Sorting & Stacking Toy Shape Color Recognition', 'Montessori wooden sorting stacking toddler educational preschool learning'),

    # Cold items (6 <= train_count <= 20)
    ('B09X1L2K3J', 19, 'LEGO City Space Mars Research Shuttle 60226 Astronaut Construction Toy', 'LEGO space shuttle mars astronaut building blocks creative toy'),
    ('B09X2M3N4P', 15, 'Buffalo Games Starry Night 500 Piece Jigsaw Puzzle Art Collection', 'Jigsaw puzzle 500 piece starry night art brain teaser tabletop'),
    ('B09X3P4Q5R', 12, 'Jellycat Bashful Beige Bunny Medium Plush Stuffed Rabbit Toy', 'Plush bunny rabbit soft stuffed animal cuddle gift toy'),
    ('B09X4R5S6T', 9, 'Montessori Busy Board for Toddlers Sensory Learning Activity Toy', 'Montessori sensory busy board wooden educational toddler learning fine motor'),
    ('B09X5T6U7V', 7, 'Ticket to Ride Europe Strategy Board Game Family Tabletop Adventure', 'Board game strategy train europe tabletop family card adventure'),
    ('B09X6V7W8X', 6, 'LEGO Botanical Collection Orchid 10311 Plant Decor Building Kit', 'LEGO botanical orchid plant building blocks creative construction'),

    # Extreme cold items (1 <= train_count <= 5)
    ('B0AX1A2B3C', 5, 'LEGO Architecture Skyline Collection Tokyo 21051 Building Set', 'LEGO architecture skyline tokyo building blocks creative construction model'),
    ('B0AX2C3D4E', 4, '3D Wooden Mechanical Puzzle Clock Kit DIY Brain Teaser Model', 'Wooden 3D jigsaw puzzle mechanical clock DIY brain teaser educational'),
    ('B0AX3E4F5G', 3, 'Aurora World Miyoni Tots Golden Retriever Puppy Plush Stuffed Dog', 'Plush golden retriever puppy dog soft stuffed animal cuddle toy'),
    ('B0AX4G5H6J', 2, 'Montessori Wooden Number Counting Peg Board Math Manipulative Toy', 'Montessori wooden counting math educational preschool toddler learning'),
    ('B0AX5J6K7L', 1, 'Azul Stained Glass Strategy Board Game Abstract Tile Placement', 'Board game strategy tile placement tabletop family puzzle'),
    ('B0AX6L7M8N', 1, 'LEGO Creator 3in1 Majestic Tiger 31129 Animal Figure Building Kit', 'LEGO creator tiger animal building blocks creative construction toy'),

    # Zero-shot items (train_count == 0)
    ('B0ZS1Q2W3E', 0, 'LEGO Icons Galaxy Explorer 10497 Classic Space Starship Building Set', 'LEGO space galaxy explorer starship astronaut building blocks construction toy'),
    ('B0ZS2E3R4T', 0, 'LEGO Ideas Tree House 21318 Botanical Cabin Building Kit', 'LEGO ideas tree house botanical cabin building blocks creative construction'),
    ('B0ZS3T4Y5U', 0, 'Galison Mudpuppy Ocean Life 1000 Piece Glowing Jigsaw Puzzle', 'Jigsaw puzzle 1000 piece ocean life brain teaser tabletop family'),
    ('B0ZS4U5I6O', 0, 'Sanrio Hello Kitty & Friends Super Soft Kawaii Plush Cushion', 'Plush hello kitty soft stuffed animal pillow kawaii cuddle toy'),
    ('B0ZS5O6P7A', 0, 'Montessori Wooden Rainbow Balance Blocks Waldorf Stacking Game', 'Montessori wooden rainbow stacking blocks educational toddler learning'),
    ('B0ZS6A7S8D', 0, 'Wingspan Bird Collection Strategy Board Game Engine Building Tabletop', 'Board game strategy bird card tabletop family adventure'),
    ('B0ZS7D8F9G', 0, 'LEGO Ninjago Lloyd Golden Ultra Dragon 71774 Action Building Set', 'LEGO ninjago dragon action building blocks creative construction toy'),
    ('B0ZS8G9H0J', 0, 'Montessori Wooden Musical Instruments Percussion Set for Kids', 'Montessori wooden musical educational toddler preschool learning toy'),
]


def build_mock_catalog(output: Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output)
    items = MOCK_ITEMS
    photo_records = []
    bundled_manifest = ROOT / 'demo/catalog_products.json'
    if bundled_manifest.is_file():
        photo_records = json.loads(bundled_manifest.read_text(encoding='utf-8'))['items']
        if len(photo_records) != 30 or len({row['asin'] for row in photo_records}) != 30:
            raise ValueError('Bundled demo catalog must contain 30 distinct products.')
        items = [(r['asin'], r['train_count'], r['title'], r['text']) for r in photo_records]
    content_dir = output / 'content'
    catalog_dir = output / 'foundation_catalog_v1'
    content_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(content_dir / 'items.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_index', 'parent_asin', 'train_count'])
        for idx, (asin, count, _, _) in enumerate(items):
            writer.writerow([idx, asin, count])

    with gzip.open(content_dir / 'products_text.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parent_asin', 'title', 'text'])
        for asin, _, title, text in items:
            writer.writerow([asin, title, f'{title}. {text}'])

    with gzip.open(catalog_dir / 'catalog.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_index', 'parent_asin', 'image_url'])
        for idx, (asin, _, _, _) in enumerate(items):
            writer.writerow([idx, asin, photo_records[idx]['image_url'] if photo_records else ''])

    if photo_records:
        local_photos = output / 'demo_images'
        local_photos.mkdir(parents=True, exist_ok=True)
        for row in photo_records:
            filename = hashlib.sha256(row['asin'].encode('utf-8')).hexdigest() + '.jpg'
            source = ROOT / 'demo/demo_images' / filename
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError(f'Bundled product photo checksum mismatch: {row["asin"]}')
            shutil.copyfile(source, local_photos / filename)

    # Build L2-normalized TF-IDF matrix over product text tokens
    vocab = {}
    docs = []
    for _, _, title, text in items:
        tokens = [w.strip('.,-&()').lower() for w in f'{title} {text}'.split() if len(w.strip('.,-&()')) >= 2]
        docs.append(tokens)
        for token in set(tokens):
            if token not in vocab:
                vocab[token] = len(vocab)

    n_docs = len(items)
    n_vocab = len(vocab)
    df = np.zeros(n_vocab, dtype=np.float32)
    tf = np.zeros((n_docs, n_vocab), dtype=np.float32)
    for row, tokens in enumerate(docs):
        for token in tokens:
            tf[row, vocab[token]] += 1.0
        for token in set(tokens):
            df[vocab[token]] += 1.0

    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf = (tfidf / norms).astype(np.float32)
    sparse.save_npz(content_dir / 'tfidf_all.npz', sparse.csr_matrix(tfidf))
    (output / 'demo_manifest.json').write_text(json.dumps({
        'products': len(items), 'local_photos': len(photo_records),
        'interaction_counts': 'synthetic demo values, not Amazon train statistics',
        'source': 'demo/catalog_products.json' if photo_records else 'built-in synthetic fallback',
    }, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = build_mock_catalog(args.output)
    print(f'Mock demo catalog ready at: {out} ({len(MOCK_ITEMS)} items)')


if __name__ == '__main__':
    main()

