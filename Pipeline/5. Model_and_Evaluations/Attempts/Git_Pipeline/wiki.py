import requests
import time
import zipfile
import sys

class WikiArticleFetcher:
    def __init__(self, keywords_manager, file_limit=None):
        """
        Initialize the WikiArticleFetcher with a KeywordsManager instance and an optional file limit.
        
        Args:
            keywords_manager (KeywordsManager): An instance of KeywordsManager to get the keywords.
            file_limit (int or None): Maximum number of files to save. Default is None (no limit).
        """
        self.keywords_manager = keywords_manager
        self.keywords = self.keywords_manager.get_keywords()  # Get the keywords from KeywordsManager
        self.file_limit = file_limit
        self.total_articles = 0
        self.total_word_count = 0
        self.start_time = None

    def search_wikipedia(self, keyword):
        """
        Search Wikipedia for articles containing a specific keyword along with "Canada" and "nuclear".

        Args:
            keyword (str): The keyword to search for.

        Returns:
            list: A list of article titles that match the search criteria.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'search',
            'format': 'json',
            'srsearch': f'"{keyword}" AND "Canada" AND "nuclear"',  # Ensure all terms are included
            'srlimit': 100  # Limit to the top 100 results for each keyword
        }
        response = requests.get(url, params=params)
        data = response.json()
        search_results = data.get('query', {}).get('search', [])
        return [result['title'] for result in search_results]

    def fetch_wikipedia_article(self, title):
        """
        Fetch the content of a Wikipedia article given its title.

        Args:
            title (str): The title of the Wikipedia article.

        Returns:
            str: The extracted text of the article.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts',
            'explaintext': True,
            'titles': title
        }
        response = requests.get(url, params=params)
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        if 'extract' in page:
            return page['extract']
        return None

    def count_words(self, text):
        """
        Count the number of words in a given text.

        Args:
            text (str): The text to count words in.

        Returns:
            int: The number of words in the text.
        """
        words = text.split()
        return len(words)

    def fetch_and_save_articles(self):
        """
        Fetch articles for all keywords and save them to text files.
        """
        self.start_time = time.time()
        file_count = 0

        for keyword in self.keywords:
            titles = self.search_wikipedia(keyword)
            for title in titles:
                if self.file_limit is not None and file_count >= self.file_limit:
                    print("File limit reached.")
                    return

                article_text = self.fetch_wikipedia_article(title)
                if article_text and "Canada" in article_text:
                    safe_title = f"wiki_{keyword}_{title.replace(' ', '_').replace('/', '_')}.txt"
                    with open(safe_title, "w", encoding="utf-8") as file:
                        file.write(article_text)
                    print(f"Saved article: {title}")
                    self.total_articles += 1
                    self.total_word_count += self.count_words(article_text)
                    file_count += 1

        end_time = time.time()
        print(f"Total articles saved: {self.total_articles}")
        print(f"Total word count: {self.total_word_count}")
        print(f"Total time taken: {end_time - self.start_time} seconds")

class FilteredWikiArticleFetcher(WikiArticleFetcher):
    def __init__(self, keywords_manager, file_limit=None, filtered_names=None):
        """
        Initialize the FilteredWikiArticleFetcher with a KeywordsManager instance, a file limit, and a list of filtered names.
        
        Args:
            keywords_manager (KeywordsManager): An instance of KeywordsManager to get the keywords.
            file_limit (int or None): Maximum number of files to save. Default is None (no limit).
            filtered_names (list): A list of article names to filter. Default is None.
        """
        super().__init__(keywords_manager, file_limit)
        self.filtered_names = filtered_names or []

    def fetch_and_save_articles(self):
        """
        Fetch articles for all keywords with filtering and save them to a ZIP file.
        """
        self.start_time = time.time()
        total_articles = 0
        zip_filename = "filtered_articles.zip"
        
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for keyword in self.keywords:
                    titles = self.search_wikipedia(keyword)
                    for title in titles:
                        if self.file_limit is not None and total_articles >= self.file_limit:
                            raise StopIteration("Article limit reached.")

                        article_text = self.fetch_wikipedia_article(title)
                        if article_text and "Canada" in article_text:
                            safe_title = f"wiki_{keyword}_{title.replace(' ', '_').replace('/', '_')}.txt"
                            if any(name in safe_title for name in self.filtered_names):
                                zipf.writestr(safe_title, article_text)
                                total_articles += 1
                                sys.stdout.write(f'\rArticles found: {total_articles}')
                                sys.stdout.flush()

        except StopIteration:
            print("\nArticle limit reached. Stopping the process.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            end_time = time.time()
            print(f"\nTotal articles found and added to ZIP: {total_articles}")
            print(f"Total time taken: {end_time - self.start_time:.2f} seconds")
