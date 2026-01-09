# Technical Implementation

## TF-IDF Analysis System (`src/tfidf.py`)

**Text Normalization Strategies:**
- **Entity Mapping Approach**: Manual entity normalization using regex patterns for consistent entity representation
- **NER Approach**: spaCy-powered Named Entity Recognition with entity standardization

**Key Features:**
- Custom stop word filtering (removes temporal terms, common reporting language)
- Entity consolidation (maps variations like "Governor Newsom", "Gavin Newsom", "Gov. Newsom" to single entities)
- Political leaning analysis (Left/Right/Neutral groupings)
- Category-based content analysis
- Configurable TF-IDF parameters with noise reduction

**Analysis Capabilities:**
- Top keyword extraction by category
- Political bias comparison across content
- Document frequency filtering
- Statistical significance validation

## Analysis Flow: Step-by-Step Process

The TF-IDF analysis follows a pipeline that transforms raw news articles into insights:

### 1. **Data Ingestion & Processing**
Load dataset, identify content categories, validate data integrity.

### 2. **Text Preprocessing**
**Entity Normalization** (dual approach):
- **Manual**: Regex-based entity mapping for consistent representation
  ```python
  'newsom_entity': ['newsom', 'gavin newsom', 'governor newsom']
  ```
- **NER**: spaCy entity recognition for PERSON, GPE, ORG standardization

**Text Cleaning**: Remove non-alphabetic characters, normalize case/whitespace, combine title and body.

### 3. **Stop Word Filtering**
Enhanced filtering for news-specific vocabulary:
```python
custom_stop_words = ENGLISH_STOP_WORDS.union({
    'said', 'according', 'report',      # Reporting language
    'week', 'monday', 'january',        # Temporal terms  
    'would', 'could', 'also'           # Common qualifiers
})
```

### 4. **TF-IDF Vectorization**
```python
TfidfVectorizer(
    max_features=1000,    # Reduce noise
    min_df=2, max_df=0.8, # Frequency filtering
    stop_words=custom_stop_words
)
```

### 5. **Analysis Execution**
- **Category Analysis**: Filter by category → preprocess → generate TF-IDF matrix → extract top keywords
- **Political Leaning** (optional): Group by Left/Right/Neutral → separate TF-IDF analysis → compare perspectives

### 6. **Output Processing**
- Clean entity markers and duplicates
- Rank by TF-IDF scores
- Generate structured results with statistical validation

**Example Output:**
```
Category: Economy → Top words: economy (0.234), inflation (0.198), budget (0.156)
Political: Left [climate, healthcare] vs Right [border, taxes]
```

## Architecture Highlights

- **Modular Design**: Separation of concerns with distinct modules for analysis and data processing
- **Dual Processing Modes**: Support for both rule-based and ML-based entity recognition
- **Error Resilience**: Exception handling and graceful degradation
- **Scalable Analysis**: Configurable parameters for different dataset sizes and analysis depths