import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import StringIO
import json
import os
from openai import OpenAI

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def get_openrouter_client():
    """Get OpenRouter client if API key is available"""
    if OPENROUTER_API_KEY:
        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return None

def ai_match_products(source_products, target_products, progress_callback=None):
    """Use AI to find matching products between two lists"""
    client = get_openrouter_client()
    if not client:
        return None
    
    matches = []
    total = len(source_products)
    
    for idx, source in enumerate(source_products):
        if progress_callback:
            progress_callback((idx + 1) / total)
        
        source_name = source.get('name', source.get('product_name', ''))
        source_brand = source.get('brand', '')
        source_model = source.get('model', '')
        source_category = source.get('category', '')
        source_desc = source.get('description', '')
        
        target_list = []
        for i, t in enumerate(target_products):
            t_name = t.get('name', t.get('product_name', ''))
            t_brand = t.get('brand', '')
            t_model = t.get('model', '')
            target_list.append(f"{i}: {t_name} (Brand: {t_brand}, Model: {t_model})")
        
        prompt = f"""You are a product matching expert. Find the BEST matching product from the target list for this source product.

SOURCE PRODUCT:
- Name: {source_name}
- Brand: {source_brand}
- Model: {source_model}
- Category: {source_category}
- Description: {source_desc[:200] if source_desc else 'N/A'}

TARGET PRODUCTS:
{chr(10).join(target_list[:50])}

INSTRUCTIONS:
1. Find products that are the SAME or very similar (same brand, model, or equivalent product)
2. Consider brand names, model numbers, product specifications
3. Return JSON with this format: {{"match_index": <number or null>, "confidence": <0-100>, "reason": "<brief explanation>"}}
4. If no good match exists, set match_index to null and confidence to 0
5. Only match if confidence is above 60%

Return ONLY valid JSON, no other text."""

        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            if result.get('match_index') is not None and result.get('confidence', 0) >= 60:
                match_idx = int(result['match_index'])
                if 0 <= match_idx < len(target_products):
                    matches.append({
                        'source_idx': idx,
                        'target_idx': match_idx,
                        'confidence': result.get('confidence', 0),
                        'reason': result.get('reason', '')
                    })
        except Exception as e:
            continue
    
    return matches

def ai_enhance_matching(source_df, target_df, similarity_threshold=60, progress_callback=None):
    """Enhanced matching using AI"""
    client = get_openrouter_client()
    if not client:
        return None
    
    source_products = source_df.to_dict('records')
    target_products = target_df.to_dict('records')
    
    ai_matches = ai_match_products(source_products, target_products, progress_callback)
    
    if not ai_matches:
        return pd.DataFrame()
    
    matches = []
    for match in ai_matches:
        source_row = source_df.iloc[match['source_idx']]
        target_row = target_df.iloc[match['target_idx']]
        
        source_name = get_product_name(source_row)
        target_name = get_product_name(target_row)
        price1 = get_price(source_row)
        price2 = get_price(target_row)
        price_diff = price2 - price1
        price_diff_pct = ((price2 - price1) / price1 * 100) if price1 > 0 else 0
        
        matches.append({
            'source_product': source_name,
            'source_price': price1,
            'source_retailer': get_retailer(source_row),
            'target_product': target_name,
            'target_price': price2,
            'target_retailer': get_retailer(target_row),
            'similarity_score': match['confidence'],
            'price_difference': round(price_diff, 2),
            'price_difference_pct': round(price_diff_pct, 1),
            'source_description': get_description(source_row),
            'target_description': get_description(target_row),
            'source_brand': source_row.get('brand', '') if 'brand' in source_row.index else '',
            'target_brand': target_row.get('brand', '') if 'brand' in target_row.index else '',
            'ai_reason': match.get('reason', '')
        })
    
    return pd.DataFrame(matches)

st.set_page_config(
    page_title="Product Matching System",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts using multiple methods"""
    ratio = fuzz.ratio(text1.lower(), text2.lower())
    partial_ratio = fuzz.partial_ratio(text1.lower(), text2.lower())
    token_sort = fuzz.token_sort_ratio(text1.lower(), text2.lower())
    token_set = fuzz.token_set_ratio(text1.lower(), text2.lower())
    
    combined_score = (ratio * 0.2 + partial_ratio * 0.2 + token_sort * 0.3 + token_set * 0.3)
    return combined_score

def find_similar_products(source_df, target_df, similarity_threshold=60):
    """Find similar products between two dataframes"""
    matches = []
    
    for idx1, row1 in source_df.iterrows():
        source_name = get_product_name(row1)
        source_desc = get_description(row1)
        source_text = f"{source_name} {source_desc}"
        source_retailer = get_retailer(row1)
        
        for idx2, row2 in target_df.iterrows():
            target_name = get_product_name(row2)
            target_desc = get_description(row2)
            target_text = f"{target_name} {target_desc}"
            target_retailer = get_retailer(row2)
            
            similarity = calculate_text_similarity(source_text, target_text)
            
            if similarity >= similarity_threshold:
                price1 = get_price(row1)
                price2 = get_price(row2)
                price_diff = price2 - price1
                price_diff_pct = ((price2 - price1) / price1 * 100) if price1 > 0 else 0
                
                matches.append({
                    'source_product': source_name,
                    'source_price': price1,
                    'source_retailer': source_retailer,
                    'target_product': target_name,
                    'target_price': price2,
                    'target_retailer': target_retailer,
                    'similarity_score': round(similarity, 1),
                    'price_difference': round(price_diff, 2),
                    'price_difference_pct': round(price_diff_pct, 1),
                    'source_description': source_desc,
                    'target_description': target_desc,
                    'source_brand': row1.get('brand', '') if 'brand' in row1.index else '',
                    'target_brand': row2.get('brand', '') if 'brand' in row2.index else '',
                })
    
    return pd.DataFrame(matches)

def create_sample_data():
    """Create sample product data for demonstration"""
    source_data = {
        'product_name': [
            'Apple iPhone 15 Pro 256GB',
            'Samsung Galaxy S24 Ultra',
            'Sony WH-1000XM5 Headphones',
            'MacBook Pro 14 inch M3',
            'Dell XPS 15 Laptop',
            'Nintendo Switch OLED',
            'iPad Pro 12.9 inch',
            'Bose QuietComfort Earbuds'
        ],
        'description': [
            'Latest iPhone with A17 Pro chip',
            'Premium Android smartphone with S Pen',
            'Wireless noise cancelling headphones',
            'Apple laptop with M3 chip',
            'Premium Windows laptop',
            'Portable gaming console',
            'Apple tablet with M2 chip',
            'True wireless earbuds'
        ],
        'price': [999, 1199, 349, 1999, 1499, 349, 1099, 279]
    }
    
    target_data = {
        'product_name': [
            'iPhone 15 Pro 256GB Black',
            'Samsung S24 Ultra 512GB',
            'Sony WH1000XM5 Black',
            'Apple MacBook Pro 14" M3 Pro',
            'Dell XPS 15 9530',
            'Nintendo Switch OLED Model',
            'Apple iPad Pro 12.9"',
            'Bose QC Earbuds II'
        ],
        'description': [
            'iPhone with titanium design',
            'Galaxy flagship phone',
            'Premium wireless headphones',
            'Professional laptop',
            'Dell premium ultrabook',
            'Gaming handheld device',
            'Professional tablet',
            'Wireless noise cancelling earbuds'
        ],
        'price': [1049, 1299, 329, 2199, 1399, 359, 1149, 249]
    }
    
    return pd.DataFrame(source_data), pd.DataFrame(target_data)

def parse_csv(uploaded_file):
    """Parse uploaded CSV file"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        return df, None
    except Exception as e:
        return None, str(e)

def parse_json(uploaded_file):
    """Parse uploaded JSON file"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        data = json.loads(content)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'products' in data:
                df = pd.DataFrame(data['products'])
            elif 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            return None, "Invalid JSON format"
        return df, None
    except Exception as e:
        return None, str(e)

def parse_file(uploaded_file):
    """Parse uploaded file (CSV or JSON)"""
    filename = uploaded_file.name.lower()
    if filename.endswith('.json'):
        return parse_json(uploaded_file)
    else:
        return parse_csv(uploaded_file)

def normalize_dataframe(df):
    """Normalize column names to standard format"""
    df = df.copy()
    
    column_mapping = {
        'name': 'product_name',
        'product': 'product_name',
        'title': 'product_name',
        'current_price': 'price',
        'sale_price': 'price',
        'selling_price': 'price',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    
    return df

def get_product_name(row):
    """Get product name from row, checking multiple possible columns"""
    if 'product_name' in row.index and pd.notna(row.get('product_name')):
        return str(row['product_name'])
    if 'name' in row.index and pd.notna(row.get('name')):
        return str(row['name'])
    return ''

def get_price(row):
    """Get price from row, checking multiple possible columns"""
    for col in ['price', 'current_price', 'sale_price', 'selling_price']:
        if col in row.index and pd.notna(row.get(col)):
            try:
                return float(row[col])
            except (ValueError, TypeError):
                continue
    return 0.0

def get_description(row):
    """Get description from row"""
    desc_parts = []
    if 'description' in row.index and pd.notna(row.get('description')):
        desc_parts.append(str(row['description']))
    if 'brand' in row.index and pd.notna(row.get('brand')):
        desc_parts.append(str(row['brand']))
    if 'model' in row.index and pd.notna(row.get('model')):
        desc_parts.append(str(row['model']))
    if 'category' in row.index and pd.notna(row.get('category')):
        desc_parts.append(str(row['category']))
    return ' '.join(desc_parts)

def get_retailer(row):
    """Get retailer from row"""
    if 'retailer' in row.index and pd.notna(row.get('retailer')):
        return str(row['retailer'])
    return ''

def main():
    st.title("🔍 Product Matching & Price Comparison")
    st.markdown("Compare products across different sources and analyze price differences")
    
    if 'source_df' not in st.session_state:
        st.session_state.source_df = None
    if 'target_df' not in st.session_state:
        st.session_state.target_df = None
    if 'matches_df' not in st.session_state:
        st.session_state.matches_df = None
    
    with st.sidebar:
        st.header("Data Input")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload Files (CSV/JSON)", "Use Sample Data", "Manual Entry"]
        )
        
        if data_source == "Use Sample Data":
            if st.button("Load Sample Data", type="primary"):
                source_df, target_df = create_sample_data()
                st.session_state.source_df = source_df
                st.session_state.target_df = target_df
                st.session_state.matches_df = None
                st.success("Sample data loaded!")
        
        elif data_source == "Upload Files (CSV/JSON)":
            st.markdown("**Supported formats:** CSV, JSON")
            st.markdown("**Required fields:** `name` or `product_name`, `current_price` or `price`")
            st.markdown("**Optional:** `description`, `brand`, `retailer`, `category`")
            
            source_file = st.file_uploader("Source Products", type=['csv', 'json'], key='source')
            target_file = st.file_uploader("Target Products", type=['csv', 'json'], key='target')
            
            if source_file and target_file:
                source_df, source_error = parse_file(source_file)
                target_df, target_error = parse_file(target_file)
                
                if source_error:
                    st.error(f"Source file error: {source_error}")
                elif target_error:
                    st.error(f"Target file error: {target_error}")
                elif source_df is not None and target_df is not None:
                    name_cols = ['product_name', 'name', 'product', 'title']
                    price_cols = ['price', 'current_price', 'sale_price', 'selling_price']
                    
                    source_has_name = any(col in source_df.columns for col in name_cols)
                    source_has_price = any(col in source_df.columns for col in price_cols)
                    target_has_name = any(col in target_df.columns for col in name_cols)
                    target_has_price = any(col in target_df.columns for col in price_cols)
                    
                    if source_has_name and source_has_price and target_has_name and target_has_price:
                        source_df = normalize_dataframe(source_df)
                        target_df = normalize_dataframe(target_df)
                        st.session_state.source_df = source_df
                        st.session_state.target_df = target_df
                        st.session_state.matches_df = None
                        st.success("Files uploaded successfully!")
                    else:
                        missing = []
                        if not source_has_name:
                            missing.append("Source: product name field")
                        if not source_has_price:
                            missing.append("Source: price field")
                        if not target_has_name:
                            missing.append("Target: product name field")
                        if not target_has_price:
                            missing.append("Target: price field")
                        st.error(f"Missing required fields: {', '.join(missing)}")
        
        elif data_source == "Manual Entry":
            st.subheader("Add to Source List")
            with st.form("source_form"):
                src_name = st.text_input("Product Name", key="src_name")
                src_desc = st.text_input("Description (optional)", key="src_desc")
                src_price = st.number_input("Price", min_value=0.0, step=0.01, key="src_price")
                
                if st.form_submit_button("Add to Source"):
                    if src_name and src_price > 0:
                        new_row = pd.DataFrame([{
                            'product_name': src_name,
                            'description': src_desc,
                            'price': src_price
                        }])
                        if st.session_state.source_df is None:
                            st.session_state.source_df = new_row
                        else:
                            st.session_state.source_df = pd.concat([st.session_state.source_df, new_row], ignore_index=True)
                        st.success("Added to source list!")
            
            st.subheader("Add to Target List")
            with st.form("target_form"):
                tgt_name = st.text_input("Product Name", key="tgt_name")
                tgt_desc = st.text_input("Description (optional)", key="tgt_desc")
                tgt_price = st.number_input("Price", min_value=0.0, step=0.01, key="tgt_price")
                
                if st.form_submit_button("Add to Target"):
                    if tgt_name and tgt_price > 0:
                        new_row = pd.DataFrame([{
                            'product_name': tgt_name,
                            'description': tgt_desc,
                            'price': tgt_price
                        }])
                        if st.session_state.target_df is None:
                            st.session_state.target_df = new_row
                        else:
                            st.session_state.target_df = pd.concat([st.session_state.target_df, new_row], ignore_index=True)
                        st.success("Added to target list!")
        
        st.divider()
        
        st.header("Matching Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold (%)",
            min_value=30,
            max_value=100,
            value=60,
            help="Minimum similarity score to consider products as matches"
        )
        
        if st.button("Clear All Data", type="secondary"):
            st.session_state.source_df = None
            st.session_state.target_df = None
            st.session_state.matches_df = None
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔗 Find Matches", "📈 Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source Products")
            if st.session_state.source_df is not None and len(st.session_state.source_df) > 0:
                st.dataframe(st.session_state.source_df, use_container_width=True, hide_index=True)
                st.metric("Total Products", len(st.session_state.source_df))
            else:
                st.info("No source products loaded. Use the sidebar to add data.")
        
        with col2:
            st.subheader("Target Products")
            if st.session_state.target_df is not None and len(st.session_state.target_df) > 0:
                st.dataframe(st.session_state.target_df, use_container_width=True, hide_index=True)
                st.metric("Total Products", len(st.session_state.target_df))
            else:
                st.info("No target products loaded. Use the sidebar to add data.")
    
    with tab2:
        st.subheader("Product Matching")
        
        if st.session_state.source_df is not None and st.session_state.target_df is not None:
            ai_available = OPENROUTER_API_KEY is not None
            
            col_method, col_btn = st.columns([2, 1])
            with col_method:
                if ai_available:
                    matching_method = st.radio(
                        "Matching Method:",
                        ["Text Similarity", "AI-Powered (OpenRouter)"],
                        horizontal=True,
                        help="AI matching uses OpenRouter to understand product semantics for better matching"
                    )
                else:
                    matching_method = "Text Similarity"
                    st.info("Add OPENROUTER_API_KEY to enable AI-powered matching")
            
            with col_btn:
                if matching_method == "AI-Powered (OpenRouter)" and ai_available:
                    run_matching = st.button("🤖 Find Matches with AI", type="primary")
                else:
                    run_matching = st.button("🔍 Find Similar Products", type="primary")
            
            if run_matching:
                if matching_method == "AI-Powered (OpenRouter)" and ai_available:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("AI is analyzing products...")
                    
                    def update_progress(pct):
                        progress_bar.progress(pct)
                        status_text.text(f"AI analyzing... {int(pct * 100)}%")
                    
                    matches_df = ai_enhance_matching(
                        st.session_state.source_df,
                        st.session_state.target_df,
                        similarity_threshold,
                        update_progress
                    )
                    progress_bar.empty()
                    status_text.empty()
                    
                    if matches_df is None or len(matches_df) == 0:
                        st.warning("No AI matches found. Try using text similarity instead.")
                    else:
                        st.session_state.matches_df = matches_df
                else:
                    with st.spinner("Analyzing products for matches..."):
                        matches_df = find_similar_products(
                            st.session_state.source_df,
                            st.session_state.target_df,
                            similarity_threshold
                        )
                        st.session_state.matches_df = matches_df
            
            if st.session_state.matches_df is not None and len(st.session_state.matches_df) > 0:
                matches_df = st.session_state.matches_df
                
                st.success(f"Found {len(matches_df)} product matches!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ["similarity_score", "price_difference", "price_difference_pct"]
                    )
                with col2:
                    sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
                with col3:
                    min_similarity = st.slider("Min Similarity", 0, 100, 0)
                
                filtered_df = matches_df[matches_df['similarity_score'] >= min_similarity]
                sorted_df = filtered_df.sort_values(
                    by=sort_by,
                    ascending=(sort_order == "Ascending")
                )
                
                for idx, row in sorted_df.iterrows():
                    source_retailer = row.get('source_retailer', '') if 'source_retailer' in row.index else ''
                    target_retailer = row.get('target_retailer', '') if 'target_retailer' in row.index else ''
                    
                    expander_title = f"🔗 {row['source_product'][:40]}... ↔ {row['target_product'][:40]}... | Match: {row['similarity_score']}%"
                    if len(row['source_product']) <= 40 and len(row['target_product']) <= 40:
                        expander_title = f"🔗 {row['source_product']} ↔ {row['target_product']} | Match: {row['similarity_score']}%"
                    
                    with st.expander(expander_title):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown("**Source Product**")
                            if source_retailer:
                                st.markdown(f"🏪 *{source_retailer}*")
                            st.write(f"📦 {row['source_product']}")
                            if 'source_brand' in row.index and row['source_brand']:
                                st.caption(f"Brand: {row['source_brand']}")
                            if row['source_description']:
                                st.caption(row['source_description'][:100])
                            st.metric("Price", f"฿{row['source_price']:,.2f}")
                        
                        with col2:
                            st.markdown("**Target Product**")
                            if target_retailer:
                                st.markdown(f"🏪 *{target_retailer}*")
                            st.write(f"📦 {row['target_product']}")
                            if 'target_brand' in row.index and row['target_brand']:
                                st.caption(f"Brand: {row['target_brand']}")
                            if row['target_description']:
                                st.caption(row['target_description'][:100])
                            st.metric("Price", f"฿{row['target_price']:,.2f}")
                        
                        with col3:
                            st.markdown("**Comparison**")
                            st.metric(
                                "Similarity",
                                f"{row['similarity_score']}%"
                            )
                            delta_color = "inverse" if row['price_difference'] > 0 else "normal"
                            st.metric(
                                "Price Diff",
                                f"฿{abs(row['price_difference']):,.2f}",
                                delta=f"{row['price_difference_pct']:+.1f}%",
                                delta_color=delta_color
                            )
                        
                        if 'ai_reason' in row.index and row['ai_reason']:
                            st.info(f"🤖 AI Reason: {row['ai_reason']}")
                
                st.divider()
                st.subheader("Matches Summary Table")
                
                display_cols = ['source_product', 'source_price', 'target_product', 'target_price', 
                               'similarity_score', 'price_difference', 'price_difference_pct']
                col_names = ['Source Product', 'Source Price (฿)', 'Target Product', 
                            'Target Price (฿)', 'Similarity %', 'Price Diff (฿)', 'Price Diff %']
                
                if 'source_retailer' in sorted_df.columns and sorted_df['source_retailer'].any():
                    display_cols.insert(1, 'source_retailer')
                    col_names.insert(1, 'Source Retailer')
                if 'target_retailer' in sorted_df.columns and sorted_df['target_retailer'].any():
                    insert_idx = display_cols.index('target_product') + 1
                    display_cols.insert(insert_idx, 'target_retailer')
                    col_names.insert(insert_idx, 'Target Retailer')
                
                display_df = sorted_df[display_cols].copy()
                display_df.columns = col_names
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="product_matches.csv",
                        mime="text/csv"
                    )
                with col_json:
                    json_data = display_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name="product_matches.json",
                        mime="application/json"
                    )
            
            elif st.session_state.matches_df is not None:
                st.warning("No matches found with the current threshold. Try lowering the similarity threshold.")
        else:
            st.info("Please load both source and target product data to find matches.")
    
    with tab3:
        st.subheader("Price Analysis")
        
        if st.session_state.matches_df is not None and len(st.session_state.matches_df) > 0:
            matches_df = st.session_state.matches_df
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Matches", len(matches_df))
            with col2:
                avg_similarity = matches_df['similarity_score'].mean()
                st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
            with col3:
                avg_price_diff = matches_df['price_difference'].mean()
                st.metric("Avg Price Diff", f"฿{avg_price_diff:,.2f}")
            with col4:
                cheaper_count = len(matches_df[matches_df['price_difference'] < 0])
                st.metric("Cheaper in Target", cheaper_count)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Price Comparison by Product")
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Source Price',
                    x=matches_df['source_product'].str[:20],
                    y=matches_df['source_price'],
                    marker_color='#1f77b4'
                ))
                
                fig.add_trace(go.Bar(
                    name='Target Price',
                    x=matches_df['source_product'].str[:20],
                    y=matches_df['target_price'],
                    marker_color='#ff7f0e'
                ))
                
                fig.update_layout(
                    barmode='group',
                    xaxis_tickangle=-45,
                    height=400,
                    margin=dict(b=120)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Price Difference Distribution")
                fig = px.histogram(
                    matches_df,
                    x='price_difference',
                    nbins=20,
                    color_discrete_sequence=['#2ecc71']
                )
                fig.update_layout(
                    xaxis_title="Price Difference (฿)",
                    yaxis_title="Count",
                    height=400
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Similarity vs Price Difference")
                fig = px.scatter(
                    matches_df,
                    x='similarity_score',
                    y='price_difference_pct',
                    size='source_price',
                    hover_data=['source_product', 'target_product'],
                    color='price_difference',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(
                    xaxis_title="Similarity Score (%)",
                    yaxis_title="Price Difference (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Similarity Score Distribution")
                fig = px.pie(
                    names=['High (80-100%)', 'Medium (60-80%)', 'Low (40-60%)', 'Very Low (<40%)'],
                    values=[
                        len(matches_df[matches_df['similarity_score'] >= 80]),
                        len(matches_df[(matches_df['similarity_score'] >= 60) & (matches_df['similarity_score'] < 80)]),
                        len(matches_df[(matches_df['similarity_score'] >= 40) & (matches_df['similarity_score'] < 60)]),
                        len(matches_df[matches_df['similarity_score'] < 40])
                    ],
                    color_discrete_sequence=['#27ae60', '#f39c12', '#e74c3c', '#95a5a6']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run product matching first to see analysis charts.")
    
    st.divider()
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        ### Getting Started
        1. **Load Data**: Use the sidebar to either:
           - Upload CSV or JSON files (source and target products)
           - Load sample data for demonstration
           - Manually enter products one by one
        
        2. **Set Threshold**: Adjust the similarity threshold to control match sensitivity
        
        3. **Find Matches**: Click the "Find Similar Products" button to identify matches
        
        4. **Analyze Results**: Review matches and price comparisons in the Analysis tab
        
        ### Supported File Formats
        
        #### CSV Format
        Your CSV files should contain these columns:
        - `name` or `product_name` (required): Name of the product
        - `current_price` or `price` (required): Product price as a number
        - `description` (optional): Product description for better matching
        - `brand` (optional): Product brand
        - `retailer` (optional): Store/retailer name
        - `category` (optional): Product category
        
        #### JSON Format
        JSON files can be structured as:
        - An array of product objects: `[{"name": "...", "current_price": 99.99}, ...]`
        - An object with a "products" key: `{"products": [...]}`
        - An object with a "data" key: `{"data": [...]}`
        
        **Example JSON structure:**
        ```json
        {
          "name": "Product Name",
          "retailer": "Store Name",
          "current_price": 169,
          "original_price": 215,
          "brand": "Brand Name",
          "category": "Category",
          "description": "Product description"
        }
        ```
        
        ### Understanding Similarity Scores
        - **80-100%**: Very high match - likely the same product
        - **60-80%**: Good match - similar products
        - **40-60%**: Moderate match - possibly related products
        - **Below 40%**: Low match - different products
        """)

if __name__ == "__main__":
    main()
