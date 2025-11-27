# Product Matching & Price Comparison System

## Overview
A Streamlit-based web application that identifies similar products between two datasets and compares their prices. The system uses text similarity algorithms to match products and provides comprehensive price analysis.

## Current State
- Fully functional MVP with product matching and price comparison features
- Sample data available for demonstration
- CSV upload and manual entry support

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
   - Sample data for quick demonstration
   - Manual product entry

2. **Product Matching**:
   - Uses RapidFuzz for fuzzy string matching
   - Combines multiple similarity metrics (ratio, partial ratio, token sort, token set)
   - Adjustable similarity threshold (30-100%)

3. **Price Comparison**:
   - Side-by-side price comparison
   - Price difference calculation (absolute and percentage)
   - Visual indicators for cheaper/expensive products

4. **Analysis Dashboard**:
   - Summary statistics (matches, avg similarity, avg price diff)
   - Price comparison bar charts
   - Price difference distribution histogram
   - Similarity vs price scatter plot
   - Similarity score distribution pie chart

5. **Export**:
   - Download matches as CSV

## Dependencies
- streamlit: Web application framework
- pandas: Data manipulation
- plotly: Interactive visualizations
- rapidfuzz: Fuzzy string matching
- scikit-learn: TF-IDF vectorization (available for future enhancements)

## CSV Format Requirements
Required columns:
- `product_name`: Name of the product
- `price`: Product price (numeric)

Optional columns:
- `description`: Product description for improved matching

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Recent Changes
- 2025-11-27: Initial implementation of product matching system with price comparison

## User Preferences
(To be updated based on user feedback)
