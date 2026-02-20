# Intelligent Amazon Search

An AI-powered Amazon product search application that uses LLMs for query understanding and intelligent reranking.

## Features

- **Query Understanding**: Automatically generates 2-3 optimized search queries from user input using Gemini Flash
- **Amazon Scraping**: Scrapes top 60 results for each generated query
- **Smart Reranking**: Uses LLM to rerank products based on relevance, quality, and purchase indicators
- **Modern Web Interface**: Clean, responsive UI with real-time search results

## Architecture

### 1. Query Understanding Module
- Takes user's natural language query
- Uses Gemini Flash to generate 2-3 optimized Amazon search queries
- Captures different aspects and variations of the search intent

### 2. Scraping Module
- Scrapes Amazon.com search results
- Collects product information: title, price, rating, reviews, images
- Handles pagination to get up to 60 results per query
- Deduplicates products across queries

### 3. Reranking Module
- Uses Gemini Flash to intelligently rerank all scraped products
- Considers:
  - Relevance to original query
  - Product quality (ratings)
  - Purchase validation (review count)
- Returns top N results with relevance scores

## Installation

1. Clone or navigate to the project directory:
```bash
cd amazon_search
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

## Configuration

Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter your search query and adjust parameters:
   - **Max results per query**: Number of products to scrape per generated query (10-100)
   - **Top results to show**: Number of reranked results to display (5-50)

4. Click "Search" and wait for results

## Project Structure

```
amazon_search/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Example environment file
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── modules/                   # Core modules
│   ├── __init__.py
│   ├── query_understanding.py # Query generation with LLM
│   ├── scraper.py             # Amazon scraping logic
│   └── reranker.py            # Product reranking with LLM
├── templates/                 # HTML templates
│   └── index.html             # Main web interface
└── static/                    # Static assets
    └── style.css              # Stylesheet
```

## How It Works

1. **User enters query** (e.g., "best wireless headphones for running")

2. **Query Understanding**:
   - LLM generates optimized queries:
     - "wireless running headphones"
     - "sports bluetooth earbuds"
     - "waterproof wireless earphones"

3. **Scraping**:
   - Scrapes up to 60 products for each query from Amazon
   - Collects ~180 total products (with deduplication)

4. **Reranking**:
   - LLM analyzes all products
   - Ranks by relevance, quality, and purchase signals
   - Returns top 20 most relevant products

5. **Display**:
   - Shows ranked products with images, prices, ratings
   - Displays relevance scores for each product

## Notes

- Amazon may implement rate limiting or blocking for scraping
- Add delays between requests to avoid detection
- Consider using proxies or rotating user agents for production use
- Web scraping should comply with Amazon's Terms of Service
- This is for educational purposes

## Technologies Used

- **Backend**: Python, Flask
- **LLM**: Google Gemini Flash (gemini-1.5-flash)
- **Scraping**: BeautifulSoup4, Requests
- **Frontend**: HTML, CSS, JavaScript (Vanilla)

## Future Enhancements

- Add caching to reduce API calls
- Implement user preferences and filters
- Support for other e-commerce sites
- Save search history
- Export results to CSV/JSON
- Add product comparison features
- Implement async scraping for better performance

## License

MIT
