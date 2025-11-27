# Product Matching & Price Comparison System

## Overview
A Streamlit-based web application that identifies similar products between two datasets and compares their prices. The system uses text similarity algorithms to match products and provides comprehensive price analysis. Optimized for Thai retail product data.

## Current State
- Fully functional MVP with product matching and price comparison features
- Sample data available for demonstration
- CSV and JSON file upload support
- Manual product entry
- Support for Thai Baht (฿) currency

## Project Structure
```
/
├── app.py              # Main Streamlit application
├── .streamlit/
│   └── config.toml     # Streamlit server configuration
├── pyproject.toml      # Python dependencies
└── replit.md           # Project documentation
```

## Key Features
1. **Data Input Methods**:
   - CSV file upload (source and target products)
   - JSON file upload (array or object with products/data key)
   - Sample data for quick demonstration
   - Manual product entry

2. **Flexible Field Mapping**:
   - Supports `name` or `product_name` for product names
   - Supports `current_price` or `price` for pricing
   - Supports additional fields: `retailer`, `brand`, `model`, `category`, `description`

3. **Product Matching**:
   - Uses RapidFuzz for fuzzy string matching
   - Combines multiple similarity metrics (ratio, partial ratio, token sort, token set)
   - Adjustable similarity threshold (30-100%)
   - Enhanced matching using brand, model, and category data

4. **Price Comparison**:
   - Side-by-side price comparison with retailer information
   - Price difference calculation (absolute and percentage)
   - Visual indicators for cheaper/expensive products
   - Thai Baht (฿) currency display

5. **Analysis Dashboard**:
   - Summary statistics (matches, avg similarity, avg price diff)
   - Price comparison bar charts
   - Price difference distribution histogram
   - Similarity vs price scatter plot
   - Similarity score distribution pie chart

6. **Export**:
   - Download matches as CSV
   - Download matches as JSON

## Dependencies
- streamlit: Web application framework
- pandas: Data manipulation
- plotly: Interactive visualizations
- rapidfuzz: Fuzzy string matching
- scikit-learn: TF-IDF vectorization (available for future enhancements)

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

### JSON Format
JSON files can be structured as:
- An array of product objects: `[{"name": "...", "current_price": 99.99}, ...]`
- An object with a "products" key: `{"products": [...]}`
- An object with a "data" key: `{"data": [...]}`

Example JSON structure:
```json
{
  "name": "กุญแจลูกปืน ISON 877C-50L 50 มม.",
  "retailer": "HomePro",
  "current_price": 169,
  "original_price": 215,
  "brand": "ISON",
  "model": "877C-50L",
  "category": "กุญแจคล้องเดี่ยว",
  "description": "วัสดุผลิตจากเหล็กคุณภาพสูง"
}
```

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Recent Changes
- 2025-11-27: Added support for flexible field names (name/current_price)
- 2025-11-27: Added retailer display and Thai Baht currency
- 2025-11-27: Enhanced matching with brand/model/category data
- 2025-11-27: Added JSON file format support for uploads and exports
- 2025-11-27: Initial implementation of product matching system with price comparison

## User Preferences
- Thai retail product data format
- Thai Baht (฿) currency display
