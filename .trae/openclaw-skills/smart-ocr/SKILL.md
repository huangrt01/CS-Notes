---
name: smart-ocr
description: Extract text from images and scanned documents using PaddleOCR - supports 100+ languages
author: claude-office-skills
version: "1.0"
tags: [ocr, paddleocr, text-extraction, multilingual, image]
models: [claude-sonnet-4, claude-opus-4]
tools: [computer, code_execution, file_operations]
library:
  name: PaddleOCR
  url: https://github.com/PaddlePaddle/PaddleOCR
  stars: 69k
---

# Smart OCR Skill

## Overview

This skill enables intelligent text extraction from images and scanned documents using **PaddleOCR** - a leading OCR engine supporting 100+ languages. Extract text from photos, screenshots, scanned PDFs, and handwritten documents with high accuracy.

## How to Use

1. Provide the image or scanned document
2. Optionally specify language(s) to detect
3. I'll extract text with position and confidence data

**Example prompts:**
- "Extract all text from this screenshot"
- "OCR this scanned PDF document"
- "Read the text from this business card photo"
- "Extract Chinese and English text from this image"

## Domain Knowledge

### PaddleOCR Fundamentals

```python
from paddleocr import PaddleOCR

# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Run OCR on image
result = ocr.ocr('image.png', cls=True)

# Result structure: [[box, (text, confidence)], ...]
for line in result[0]:
    box = line[0]      # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text = line[1][0]  # Extracted text
    conf = line[1][1]  # Confidence score
    print(f"{text} ({conf:.2f})")
```

### Supported Languages

```python
# Common language codes
languages = {
    'en': 'English',
    'ch': 'Chinese (Simplified)',
    'cht': 'Chinese (Traditional)',
    'japan': 'Japanese',
    'korean': 'Korean',
    'french': 'French',
    'german': 'German',
    'spanish': 'Spanish',
    'russian': 'Russian',
    'arabic': 'Arabic',
    'hindi': 'Hindi',
    'vi': 'Vietnamese',
    'th': 'Thai',
    # ... 100+ languages supported
}

# Use specific language
ocr = PaddleOCR(lang='ch')  # Chinese
ocr = PaddleOCR(lang='japan')  # Japanese
ocr = PaddleOCR(lang='multilingual')  # Auto-detect
```

### Configuration Options

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    # Detection settings
    det_model_dir=None,         # Custom detection model
    det_limit_side_len=960,     # Max side length for detection
    det_db_thresh=0.3,          # Binarization threshold
    det_db_box_thresh=0.5,      # Box score threshold
    
    # Recognition settings
    rec_model_dir=None,         # Custom recognition model
    rec_char_dict_path=None,    # Custom character dictionary
    
    # Angle classification
    use_angle_cls=True,         # Enable angle classification
    cls_model_dir=None,         # Custom classification model
    
    # Language
    lang='en',                  # Language code
    
    # Performance
    use_gpu=True,               # Use GPU if available
    gpu_mem=500,                # GPU memory limit (MB)
    enable_mkldnn=True,         # CPU optimization
    
    # Output
    show_log=False,             # Suppress logs
)
```

### Processing Different Sources

#### Image Files
```python
# Single image
result = ocr.ocr('image.png')

# Multiple images
images = ['img1.png', 'img2.png', 'img3.png']
for img in images:
    result = ocr.ocr(img)
    process_result(result)
```

#### PDF Files (Scanned)
```python
from pdf2image import convert_from_path

def ocr_pdf(pdf_path):
    """OCR a scanned PDF."""
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)
    
    all_text = []
    for i, img in enumerate(images):
        # Save temp image
        temp_path = f'temp_page_{i}.png'
        img.save(temp_path)
        
        # OCR the image
        result = ocr.ocr(temp_path)
        
        # Extract text
        page_text = '\n'.join([line[1][0] for line in result[0]])
        all_text.append(f"--- Page {i+1} ---\n{page_text}")
        
        os.remove(temp_path)
    
    return '\n\n'.join(all_text)
```

#### URLs and Bytes
```python
import requests
from io import BytesIO

# From URL
response = requests.get('https://example.com/image.png')
result = ocr.ocr(BytesIO(response.content))

# From bytes
with open('image.png', 'rb') as f:
    img_bytes = f.read()
result = ocr.ocr(BytesIO(img_bytes))
```

### Result Processing

```python
def process_ocr_result(result):
    """Process OCR result into structured data."""
    
    lines = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]
        
        # Calculate bounding box
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        
        lines.append({
            'text': text,
            'confidence': confidence,
            'bbox': {
                'left': min(x_coords),
                'top': min(y_coords),
                'right': max(x_coords),
                'bottom': max(y_coords),
            },
            'raw_box': box
        })
    
    return lines

# Sort by position (top to bottom, left to right)
def sort_by_position(lines):
    return sorted(lines, key=lambda x: (x['bbox']['top'], x['bbox']['left']))
```

### Text Layout Reconstruction

```python
def reconstruct_layout(result, line_threshold=10):
    """Reconstruct text layout from OCR results."""
    
    lines = process_ocr_result(result)
    lines = sort_by_position(lines)
    
    # Group into logical lines
    text_lines = []
    current_line = []
    current_y = None
    
    for line in lines:
        y = line['bbox']['top']
        
        if current_y is None or abs(y - current_y) < line_threshold:
            current_line.append(line)
            current_y = y
        else:
            # New line
            text_lines.append(' '.join([l['text'] for l in current_line]))
            current_line = [line]
            current_y = y
    
    # Add last line
    if current_line:
        text_lines.append(' '.join([l['text'] for l in current_line]))
    
    return '\n'.join(text_lines)
```

## Best Practices

1. **Preprocess Images**: Improve quality before OCR
2. **Choose Correct Language**: Specify language for better accuracy
3. **Handle Multi-column**: Process columns separately
4. **Filter Low Confidence**: Skip results below threshold
5. **Batch Processing**: Process multiple images efficiently

## Common Patterns

### Image Preprocessing
```python
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image_path):
    """Preprocess image for better OCR."""
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    # Save preprocessed
    preprocessed_path = 'preprocessed.png'
    img.save(preprocessed_path)
    
    return preprocessed_path
```

### Batch OCR with Progress
```python
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def batch_ocr(image_paths, max_workers=4):
    """OCR multiple images in parallel."""
    
    results = {}
    
    def process_single(img_path):
        result = ocr.ocr(img_path)
        return img_path, result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single, p) for p in image_paths]
        
        for future in tqdm(futures, desc="Processing OCR"):
            path, result = future.result()
            results[path] = result
    
    return results
```

## Examples

### Example 1: Business Card Reader
```python
from paddleocr import PaddleOCR
import re

def read_business_card(image_path):
    """Extract contact info from business card."""
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image_path)
    
    # Extract all text
    all_text = []
    for line in result[0]:
        all_text.append(line[1][0])
    
    full_text = '\n'.join(all_text)
    
    # Parse contact info
    contact = {
        'name': None,
        'email': None,
        'phone': None,
        'company': None,
        'title': None,
        'raw_text': full_text
    }
    
    # Email pattern
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
    if email_match:
        contact['email'] = email_match.group()
    
    # Phone pattern
    phone_match = re.search(r'[\+\d][\d\s\-\(\)]{8,}', full_text)
    if phone_match:
        contact['phone'] = phone_match.group().strip()
    
    # Name is usually the largest/first text
    if all_text:
        contact['name'] = all_text[0]
    
    return contact

card_info = read_business_card('business_card.jpg')
print(f"Name: {card_info['name']}")
print(f"Email: {card_info['email']}")
print(f"Phone: {card_info['phone']}")
```

### Example 2: Receipt Scanner
```python
from paddleocr import PaddleOCR
import re

def scan_receipt(image_path):
    """Extract items and total from receipt."""
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image_path)
    
    lines = []
    for line in result[0]:
        text = line[1][0]
        y_pos = line[0][0][1]
        lines.append({'text': text, 'y': y_pos})
    
    # Sort by vertical position
    lines.sort(key=lambda x: x['y'])
    
    receipt = {
        'items': [],
        'subtotal': None,
        'tax': None,
        'total': None
    }
    
    for line in lines:
        text = line['text']
        
        # Look for total
        if 'total' in text.lower():
            amount = re.search(r'\$?([\d,]+\.?\d*)', text)
            if amount:
                if 'sub' in text.lower():
                    receipt['subtotal'] = float(amount.group(1).replace(',', ''))
                else:
                    receipt['total'] = float(amount.group(1).replace(',', ''))
        
        # Look for tax
        elif 'tax' in text.lower():
            amount = re.search(r'\$?([\d,]+\.?\d*)', text)
            if amount:
                receipt['tax'] = float(amount.group(1).replace(',', ''))
        
        # Look for items (line with price)
        else:
            item_match = re.search(r'(.+?)\s+\$?([\d,]+\.?\d+)$', text)
            if item_match:
                receipt['items'].append({
                    'name': item_match.group(1).strip(),
                    'price': float(item_match.group(2).replace(',', ''))
                })
    
    return receipt

receipt_data = scan_receipt('receipt.jpg')
print(f"Items: {len(receipt_data['items'])}")
print(f"Total: ${receipt_data['total']}")
```

### Example 3: Multi-language Document
```python
from paddleocr import PaddleOCR

def ocr_multilingual(image_path, languages=['en', 'ch']):
    """OCR document with multiple languages."""
    
    all_results = {}
    
    for lang in languages:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        result = ocr.ocr(image_path)
        
        texts = []
        for line in result[0]:
            texts.append({
                'text': line[1][0],
                'confidence': line[1][1]
            })
        
        all_results[lang] = texts
    
    # Merge results, keeping highest confidence
    merged = {}
    for lang, texts in all_results.items():
        for item in texts:
            text = item['text']
            conf = item['confidence']
            
            if text not in merged or merged[text]['confidence'] < conf:
                merged[text] = {'confidence': conf, 'language': lang}
    
    return merged

result = ocr_multilingual('bilingual_document.png')
for text, info in result.items():
    print(f"[{info['language']}] {text} ({info['confidence']:.2f})")
```

## Limitations

- Handwritten text accuracy varies
- Very small text may not be detected
- Complex backgrounds reduce accuracy
- Rotated text needs angle classification
- GPU recommended for best performance

## Installation

```bash
# CPU version
pip install paddlepaddle paddleocr

# GPU version (CUDA 11.x)
pip install paddlepaddle-gpu paddleocr

# Additional dependencies
pip install pdf2image Pillow
```

## Resources

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md)
- [Multi-language Support](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/multi_languages_en.md)
