# Product Matching & Price Comparison System

## Overview
A Streamlit-based web application for identifying similar products and comparing prices across different retailers, specifically optimized for Thai retail product data. The system employs weighted attribute matching, AI-powered matching using Gemini models via OpenRouter, and visual similarity analysis through image matching. Its primary purpose is to help users find identical products and functional equivalents (house brands) to compare prices and discover alternatives. The project aims for high accuracy in product matching across multiple retailers, with a business vision to provide a robust tool for competitive analysis and smart shopping decisions in the retail sector.

## User Preferences
- Thai retail product data format
- Thai Baht (฿) currency display
- OpenRouter API for AI and vision matching
- Weighted attribute scoring for accuracy

## System Architecture
The system is built as a Streamlit web application, featuring two main matching modes:
1.  **Price Match Engine**: Focuses on matching identical products (same brand, model, specs).
2.  **House Brand Matcher**: Identifies functional equivalents across different brands, considering similar specifications and price tolerance.

**UI/UX Decisions:**
-   Streamlit framework provides a responsive and interactive user interface.
-   Visual indicators for price differences and comprehensive analysis dashboards (bar charts, histograms, scatter plots) for clear data presentation.
-   Side-by-side product image display for visual comparison.

**Technical Implementations & Feature Specifications:**
-   **Data Input**: Supports CSV and JSON file uploads, manual product entry, and includes sample data for demonstration. Flexible field mapping accommodates various column names for product attributes (e.g., `name`/`product_name`, `current_price`/`price`, multiple image URL fields).
-   **Weighted Attribute Matching**: A core component, assigning weights to attributes like Product Name (25%), Brand (20%), Model (20%), Dimensions (12%), Category (8%), Material (5%), Color (5%), Description (3%), and Images (2%). This system normalizes calculations for missing attributes.
-   **AI-Powered Matching**: Integrates OpenRouter's `google/gemini-2.5-flash-lite` model. It uses a hybrid approach: text pre-filtering to narrow down candidates, followed by AI analysis on the top 10 to reduce API calls while maintaining semantic accuracy.
-   **Image Matching**: Utilizes Gemini 2.5 Flash vision API for visual similarity analysis, assigning a 0-100% similarity score. This score is integrated into the weighted attribute matching with a 2% weight. The system includes graceful fallback for unavailable images.
-   **Price Comparison**: Calculates absolute and percentage price differences, displays prices in Thai Baht (฿), and offers sorting/filtering options.
-   **Persistence & Export**: Automatically saves results to timestamped JSON files and can restore previous sessions. Supports exporting matches to CSV and JSON formats.
-   **Product Line Distinction Rules**: Implements specific rules to differentiate distinct product lines (e.g., JOTASHIELD vs. JOTASHIELD FLEX) and handles Thai-English product name normalization (e.g., VINILEX = วีนิเลกซ์).
-   **House Brand Matching Criteria**: Requires products to serve the same function, be from different brands, have similar specifications, and fall within a configurable price tolerance (default 30%).

**System Design Choices:**
-   **Modularity**: Separated `price_match_engine` and `house_brand_engine` into distinct applications.
-   **Test-driven Approach**: Includes a `tests/` directory with scripts for accuracy validation across retailers and different matching modes.
-   **Configuration**: Uses `.streamlit/config.toml` for server configuration and `pyproject.toml` for Python dependency management.
-   **Robustness**: Features like retry mechanisms, category compatibility checks, targeted conflict blocking, and validation rules enhance matching accuracy and system stability.

## External Dependencies
-   **OpenRouter API**: Used for AI-powered text matching (google/gemini-2.5-flash-lite) and image matching (Gemini 2.5 Flash Vision API). Requires `OPENROUTER_API_KEY`.
-   **streamlit**: Web application framework.
-   **pandas**: For data manipulation.
-   **plotly**: For interactive data visualizations.
-   **rapidfuzz**: For fuzzy string matching.
-   **scikit-learn**: For TF-IDF vectorization.
-   **openai**: Used as the client for OpenRouter API interactions (OpenAI-compatible).