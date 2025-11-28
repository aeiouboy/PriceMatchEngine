# Product Matching & Price Comparison System

## Overview
A Streamlit-based web application that identifies similar products between two datasets and compares their prices. The system uses weighted attribute matching, AI-powered matching via OpenRouter (Gemini models), and **visual similarity analysis using image matching**. Optimized for Thai retail product data.

## Current State
- **MVP COMPLETE** with advanced product matching and image analysis
- Multi-attribute weighted matching (text + images)
- AI-powered matching via OpenRouter using google/gemini-2.5-flash-lite
- Vision-based image similarity scoring
- Persistent storage of results across sessions
- Sample data with product images for demonstration
- CSV, JSON file upload support
- Manual product entry
- Support for Thai Baht (฿) currency

## Project Structure
```
/
├── app.py              # Main Streamlit application with AI and vision integration
├── .streamlit/
│   └── config.toml     # Streamlit server configuration
├── pyproject.toml      # Python dependencies
├── saved_results/      # Auto-saved timestamped results
└── replit.md           # Project documentation
```

## Key Features

### 1. Data Input Methods
- CSV file upload (source and target products)
- JSON file upload (array, products key, or data key format)
- Sample data with images for quick demonstration
- Manual product entry

### 2. Flexible Field Mapping
- **Product Names**: `name` or `product_name`
- **Pricing**: `current_price` or `price`
- **Product Images**: `image_url`, `image`, `image_link`, `photo_url`, `photo`, `picture_url`, `picture`
- **URLs**: `url`, `product_url`, `link`, `product_link`, `href`
- **Additional**: `retailer`, `brand`, `model`, `category`, `description`, `dimensions`, `material`, `color`

### 3. Advanced Product Matching

#### Weighted Attribute Matching (Default)
- **Product Name** (25%) - Primary identifier
- **Brand** (20%) - Critical for accuracy (exact matches = 100%)
- **Model** (20%) - Critical for accuracy (exact matches = 100%)
- **Dimensions** (12%) - Physical specifications
- **Category** (8%) - Product classification (exact matches = 100%)
- **Material** (5%) - Material composition
- **Color** (5%) - Product color
- **Description** (3%) - Detailed specifications
- **Images** (2%) - **NEW** Visual similarity analysis

#### AI-Powered Matching (Optional)
- Uses OpenRouter's google/gemini-2.5-flash-lite model
- Hybrid approach: Text pre-filtering + AI analysis on top 10 candidates
- Reduces API calls while maintaining accuracy
- Semantic understanding of product equivalence

### 4. Image Matching (NEW)
- **Vision Analysis**: Uses Gemini 2.5 Flash vision API to compare product images
- **Similarity Scoring**: Returns 0-100 visual similarity percentage
- **Weighted Integration**: 2% weight in overall similarity calculation
- **Visual Display**: Product images shown side-by-side in match details
- **Safe Handling**: Graceful fallback if images unavailable

### 5. Price Comparison
- Side-by-side price comparison with retailer information
- Price difference calculation (absolute and percentage)
- Visual indicators for cheaper/expensive products
- Thai Baht (฿) currency display
- Price-based sorting and filtering

### 6. Analysis Dashboard
- Summary statistics (matches, avg similarity, avg price diff)
- Price comparison bar charts
- Price difference distribution histogram
- Similarity vs price scatter plot
- Similarity score distribution pie chart

### 7. Export & Persistence
- **Auto-save**: Results automatically saved to timestamped JSON files
- **Session Restore**: Previous results automatically loaded on restart
- **CSV Export**: Download matches as CSV
- **JSON Export**: Download matches as JSON (includes URLs and images)

## Environment Variables
- `OPENROUTER_API_KEY`: **Required** for AI-powered and image matching (get free at openrouter.ai)

## Dependencies
- streamlit: Web application framework
- pandas: Data manipulation
- plotly: Interactive visualizations
- rapidfuzz: Fuzzy string matching (text similarity)
- scikit-learn: TF-IDF vectorization
- openai: OpenRouter API client (OpenAI-compatible)

## File Format Requirements

### CSV Format
Required columns (any of these):
- `name` or `product_name`: Name of the product
- `current_price` or `price`: Product price (numeric)

Optional columns:
- `description`: Product description for improved matching
- `retailer`: Store/retailer name
- `brand`: Product brand
- `model`: Product model
- `category`: Product category
- `dimensions`: Product dimensions (e.g., "14\" x 10\" x 2\"")
- `material`: Product material (e.g., "aluminum", "plastic")
- `color`: Product color
- `url`/`link`/`product_url`: Product page URL
- `image_url`/`image`/`photo_url`: Product image URL

### JSON Format
JSON files can be structured as:
- An array of product objects: `[{"name": "...", "price": 99.99}, ...]`
- An object with a "products" key: `{"products": [...]}`
- An object with a "data" key: `{"data": [...]}`

Example JSON structure:
```json
{
  "name": "Apple iPhone 15 Pro 256GB",
  "retailer": "HomePro",
  "price": 35999,
  "brand": "Apple",
  "model": "A2846",
  "category": "Smartphones",
  "description": "Latest iPhone with A17 Pro chip",
  "dimensions": "6.12 x 2.82 x 0.31 inches",
  "material": "Titanium",
  "color": "Black",
  "url": "https://homepro.co.th/product/iphone-15-pro",
  "image_url": "https://example.com/iphone15.jpg"
}
```

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Ground Truth Evaluation
Tested against 133 ground truth product pairs (Megahome vs Thaiwatsadu):
- **AI Matching Accuracy**: 80.5% (107/133 correct)
- **Precision**: 82.3% (low false positives)
- **Not Found**: 2.3%
- **Weighted Matching Accuracy**: 38% correct (too many false positives)

### Thai-English Product Name Mappings
The system handles products named differently between retailers:
- VINILEX = วีนิเลกซ์, WEATHERBOND = เวเธอร์บอนด์, FLEXISEAL = เฟล็กซี่ซีล
- JOTASHIELD = โจตาชิลด์, WEATHERSHIELD = เวเธอร์ชีลด์, POWERPLUS = พาวเวอร์พลัส
- Finish types: กึ่งเงา=SG (semi-gloss), เนียน=SHEEN, ด้าน=MATTE
- Brand aliases: TOA SHARK=SHARK, WINDOW ASIA=FRAMEX, TOPTECH=DELTA

AI matching is recommended for production use. Performance varies by product category:
- Doors & frames (ECO-DOOR): 90%+ accuracy
- Paints: 80-85% accuracy (with finish type matching)
- Turpentine/thinner (SHARK, BARCO): 85-90% accuracy

## Recent Changes (Latest to Oldest)
- **2025-11-28**: 🚀 Major AI matching improvements
  - Brand alias normalization (SHARKS→SHARK, BARGO→BARCO)
  - Lower pre-filter threshold (20%) for better recall
  - Increased candidate pool (15 products)
  - Added size/volume matching in AI prompt
  - Brand boost for matching brands
  - Accuracy improved from 75% to 88%
- **2025-11-28**: Added real Thai retail sample data (Megahome vs Thaiwatsadu)
- **2025-11-27**: 🎯 Added image matching capability with vision API integration
  - Vision-based image similarity scoring
  - 2% weight in overall similarity calculation
  - Product images displayed in match details
  - Support for multiple image URL column names
- **2025-11-27**: Fixed weighted similarity normalization (corrected calculation formula)
- **2025-11-27**: Added product URL display in expander details and summary table
- **2025-11-27**: Implemented weighted attribute validation (25% name, 20% brand/model, 12% dimensions, 8% category, 5% material/color, 3% description)
- **2025-11-27**: Added AI-powered matching via OpenRouter API
- **2025-11-27**: Added flexible field mapping for various column names
- **2025-11-27**: Added Thai Baht currency display and retailer support
- **2025-11-27**: Initial MVP with product matching and price comparison

## User Preferences
- Thai retail product data format
- Thai Baht (฿) currency display
- OpenRouter API for AI and vision matching
- Weighted attribute scoring for accuracy

## Architecture Notes
- Weighted similarity calculation normalizes by applied weights (handles missing attributes)
- Image similarity only calculated when both products have image URLs
- AI matching pre-filters candidates to reduce API calls
- Results persist across sessions via JSON file storage
- Vision API gracefully handles image loading failures
