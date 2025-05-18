import streamlit as st
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import openai
import os
from dotenv import load_dotenv
from newspaper import Article
from html import escape
import logging
import time
from tldextract import extract

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Cache decorator for search results
@st.cache_data
def cached_search(query, num_results=4):
    """Cache search results to avoid repeated calls"""
    return list(search(query, num_results=num_results))

# Decorator for rate limiting
@st.cache_data
def rate_limited_api_call(func, *args, **kwargs):
    """Rate limit API calls"""
    time.sleep(1)  # 1 second delay between calls
    return func(*args, **kwargs)

class WordPressAuto:
    def __init__(self):
        self.search_results = []
        self.processed_text = ""
        self.generated_article = ""
        self.source_urls = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.progress_bar = None
        self.originality_score = 0.0
        self.meta_tags = None
        self.safety_check = None
        self.language = "English"

    def fetch_google_results(self, query):
        """Fetch top 4 Google search results with caching and rate limiting"""
        try:
            self.progress_bar = st.progress(0)
            self.progress_bar.progress(25)
            
            # Use cached search
            results = cached_search(query, num_results=4)
            self.search_results = []
            self.source_urls = []
            
            # Process results
            for result in results:
                if result.url not in self.source_urls:
                    self.source_urls.append(result.url)
                    
            if not self.source_urls:
                raise ValueError("No results found or Google blocked the request")
                
            return True
            
        except Exception as e:
            logging.error(f"Error fetching Google results: {str(e)}")
            return False

    def extract_content(self, url):
        """Extract content from a URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from main content
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
            text = ' '.join([p.get_text() for p in paragraphs])
            return text
        except:
            return ""

    def process_content(self):
        """Process and combine content from all sources"""
        all_text = []
        
        for url in self.source_urls:
            content = self.extract_content(url)
            if content:
                all_text.append(content)
        
        self.processed_text = ' '.join(all_text)
        return self.processed_text

    def generate_article(self):
        """Generate a high-quality article using GPT with safety and originality checks"""
        try:
            self.progress_bar.progress(50)
            
            # Add language preference to prompt
            prompt = f"""Please write a high-quality, journalistic article based on the following content:
            {self.processed_text}
            
            Requirements:
            1. Write in a professional, journalistic style
            2. Make the content unique and engaging
            3. Include relevant quotes and statistics where appropriate
            4. Structure the article with proper headings and subheadings
            5. Make it SEO-friendly while maintaining natural language
            6. Keep the tone informative and objective
            7. Add your own analysis and insights where relevant
            8. Write in {self.language} language
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional journalist and content writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            self.generated_article = response.choices[0].message.content
            
            # Calculate originality score
            source_texts = [self.extract_content(url) for url in self.source_urls]
            self.originality_score = self.calculate_originality_score(self.generated_article, source_texts)
            
            # Generate meta tags
            self.meta_tags = self.generate_meta_tags()
            
            # Check content safety
            self.safety_check = self.check_content_safety(self.generated_article)
            
            self.progress_bar.progress(75)
            return True
            
        except Exception as e:
            logging.error(f"Error generating article: {str(e)}")
            return False

    def analyze_seo(self):
        """Comprehensive SEO analysis with safety checks"""
        try:
            # 1. Keyword Analysis
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            tfidf_matrix = vectorizer.fit_transform([self.generated_article])
            feature_names = vectorizer.get_feature_names_out()
            sorted_keywords = [feature_names[i] for i in tfidf_matrix.toarray()[0].argsort()[::-1]]
            
            # 2. Readability Score
            word_count = len(self.generated_article.split())
            sentence_count = len(nltk.sent_tokenize(self.generated_article))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # 3. Keyword Density
            word_freq = nltk.FreqDist(nltk.word_tokenize(self.generated_article.lower()))
            total_words = sum(word_freq.values())
            keyword_density = {word: (count/total_words)*100 for word, count in word_freq.items()}
            
            # 4. Safety Score
            safety_score = 1.0 if not self.safety_check['flagged'] else 0.0
            
            # 5. Originality Score
            originality_score = self.originality_score
            
            return {
                'top_keywords': sorted_keywords[:10],
                'word_count': word_count,
                'avg_sentence_length': round(avg_sentence_length, 1),
                'keyword_density': {k: round(v, 2) for k, v in sorted(
                    keyword_density.items(), 
                    key=lambda item: item[1], 
                    reverse=True)[:10]},
                'meta_tags': self.meta_tags,
                'safety_score': safety_score,
                'originality_score': originality_score,
                'meta_description': self.generate_meta_description()
            }
            
        except Exception as e:
            logging.error(f"Error in SEO analysis: {str(e)}")
            return {}

    def calculate_originality_score(self, generated_article, source_texts):
        # Calculate cosine similarity between generated article and source texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([generated_article] + source_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        similarity_scores = similarity_matrix.flatten()
        originality_score = 1 - max(similarity_scores)
        return originality_score

    def generate_meta_tags(self):
        # Generate meta tags based on top keywords
        top_keywords = self.analyze_seo()['top_keywords']
        meta_tags = ', '.join(top_keywords)
        return meta_tags

    def check_content_safety(self, generated_article):
        # Check content safety using OpenAI's content filter
        response = openai.Moderation.create(
            input=generated_article,
            model="text-moderation-latest"
        )
        safety_check = response.results[0]
        return safety_check

    def generate_meta_description(self):
        # Generate meta description based on generated article
        meta_description = self.generated_article[:155]
        return meta_description

def main():
    st.title("WordPress Auto - AI-Powered Article Generator")
    
    # Initialize the app
    app = WordPressAuto()
    
    # Input form
    query = st.text_input("Enter your search query:")
    
    if st.button("Generate Article"):
        if not query:
            st.error("Please enter a search query")
            return
            
        with st.spinner("Fetching Google results..."):
            if not app.fetch_google_results(query):
                st.error("Failed to fetch Google results")
                return
                
        with st.spinner("Processing content..."):
            processed_text = app.process_content()
            
        with st.spinner("Generating article..."):
            if not app.generate_article():
                st.error("Failed to generate article")
                return
                
        with st.spinner("Analyzing SEO..."):
            seo_keywords = app.analyze_seo()
            
        # Display results
        st.success("Article generated successfully!")
        
        st.header("Generated Article")
        st.markdown(app.generated_article)
        
        st.header("Sources Used")
        for url in app.source_urls:
            st.markdown(f"- [{url}]({url})")
            
        st.header("SEO Analysis")
        st.write("Important keywords found in the article:")
        st.write(', '.join(seo_keywords))

if __name__ == "__main__":
    main()
