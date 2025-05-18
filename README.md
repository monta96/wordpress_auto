# WordPress Auto - AI-Powered Article Generator

An intelligent application that generates high-quality articles based on Google search results. It automatically:
1. Fetches top 4 Google search results for your query
2. Extracts and combines relevant content
3. Generates a unique, professional article using AI
4. Provides comprehensive SEO analysis
5. Checks content safety and originality
6. Generates meta tags and descriptions

## Features
- AI-powered content generation using GPT-3.5
- Multi-language support (English, Spanish, French)
- SEO optimization with:
  - Keyword analysis
  - Originality scoring
  - Content safety checks
  - Meta tag generation
- Source tracking and citation
- Professional journalistic style
- Content preview mode
- Progress tracking
- Downloadable articles
- Detailed article statistics

## Setup
1. Install Python 3.8 or higher
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage
1. Run the application:
   ```
   python -m streamlit run app.py
   ```
2. In the web interface:
   - Enter your search query
   - Select preferred article language
   - Adjust article settings (tone, length, creativity)
   - Click "Generate Article"
   - Monitor progress with the progress bar
   - View generated article and SEO analysis
   - Download article if desired

## Requirements
- Python 3.8+
- OpenAI API key
- Stable internet connection
- Required packages (listed in requirements.txt)

## Project Structure
```
wordpress_auto/
├── app.py          # Main application file
├── requirements.txt # Python dependencies
├── .env            # Environment variables
├── .gitignore      # Git ignore file
└── README.md       # This file
```

## Security
- Content safety checks using OpenAI's moderation API
- Rate limiting for API calls
- Environment variables for sensitive data
- Logging for debugging

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Note
This application uses OpenAI's GPT model to generate content. Make sure to:
1. Review and edit the generated content before publishing
2. Verify sources and accuracy of information
3. Follow copyright and fair use guidelines
4. Use the content for educational purposes only

## Support
For support, please open an issue in the GitHub repository.
