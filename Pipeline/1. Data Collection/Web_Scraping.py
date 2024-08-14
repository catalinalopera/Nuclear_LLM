# %%
!pip install -q requests boto3 google-cloud-storage

# %%
class KeywordsManager:
    def __init__(self, keywords=None):
        """
        Initialize the KeywordsManager with a list of keywords.

        Args:
            keywords (list): A list of keywords. If None, uses default keywords.
        """
        if keywords is None:
            self.keywords = [
                "nuclear safety", "nuclear security", "nuclear regulations", "nuclear industry",
                "nuclear act", "Canada Energy Regulator", "nuclear facility", "nuclear facilities",
                "CNSC", "Nuclear Safety and Control Act", "Canadian Nuclear Safety Commission",
                "CNSC regulatory documents", "Nuclear Facilities Regulations",
                "International Atomic Energy Agency", "IAEA Regulations", "IAEA", "IAEA Safety Glossary",
                "certification of prescribed nuclear equipment", "REGDOC", "RegDoc",
                "nuclear safety standards", "nuclear reactor safety", "radiation protection",
                "nuclear safety culture", "nuclear safety regulations", "nuclear plant safety",
                "nuclear safety analysis", "emergency preparedness nuclear", "nuclear safety protocols",
                "nuclear accident prevention", "safety of nuclear facilities", "nuclear safety management",
                "nuclear risk assessment", "nuclear safety engineering", "nuclear safety guidelines",
                "nuclear regulatory framework", "nuclear regulations compliance", "nuclear safety laws",
                "nuclear regulatory authority", "nuclear industry regulations", "nuclear regulatory standards",
                "nuclear licensing regulations", "nuclear regulatory policies", "nuclear security regulations",
                "nuclear regulatory compliance", "regulatory oversight nuclear", "nuclear energy regulation",
                "nuclear material regulations", "nuclear environmental regulations", "nuclear waste regulations",
                "nuclear security standards", "nuclear facility security", "nuclear security measures",
                "nuclear material security", "nuclear security regulations", "nuclear security protocols",
                "nuclear security threats", "nuclear security compliance", "nuclear security policies",
                "nuclear security frameworks", "nuclear security technology", "nuclear security law",
                "nuclear security incidents", "nuclear security assessments", "nuclear security strategy",
                "security of nuclear substances", "nuclear fission", "nuclear fusion", "radioactive decay",
                "half-life", "critical mass", "nuclear chain reaction", "neutron moderation", "nuclear reactor",
                "control rods", "nuclear fuel cycle", "radioactive waste management", "nuclear radiation",
                "alpha particles", "beta particles", "gamma rays", "neutron flux", "nuclear isotopes",
                "radioactive contamination", "nuclear meltdown", "radiation shielding", "nuclear power plant",
                "uranium enrichment", "plutonium reprocessing", "nuclear decommissioning", "nuclear proliferation",
                "nuclear safeguards", "radiation dosimetry", "thermal neutron", "fast neutron", "breeder reactor",
                "Atomic Energy of Canada", "nuclear material", "radiation protection", "code of practice",
                "REGDOC-3.6", "Atomic Energy of Canada Limited", "authorized nuclear operator",
                "boiling water reactor", "Canada Deuterium Uranium", "criticality accident sequence assessment",
                "Canadian Council of Ministers of the Environment", "Canadian Environmental Assessment Act",
                "certified exposure device operator", "Canadian Environmental Protection Act", "counterfeit",
                "curie", "Canadian Nuclear Safety Commission", "criticality safety control",
                "emergency core cooling system", "extended loss of AC power", "Federal Nuclear Emergency Plan",
                "fitness for duty", "fuel incident notification and analysis system", "gigabecquerel", "gray",
                "high-enriched uranium", "hydrogenated tritium oxide", "International Atomic Energy Agency",
                "irradiated fuel bay", "Institute of Nuclear Power Operations", "International Physical Protection Advisory Service",
                "International Reporting System for Operating Experience", "International Nuclear and Radiological Event Scale",
                "International Commission on Radiological Protection", "International Commission on Radiation Units and Measurements",
                "low-enriched uranium", "loss-of-coolant accident", "megabecquerel", "micro modular reactor",
                "nuclear criticality safety", "National Non-Destructive Testing Certification Body", "nuclear emergency management",
                "Nuclear Emergency Organization", "nuclear energy worker", "Nuclear Suppliers Group", "spent nuclear fuel",
                "safe operating envelope", "sievert", "International System of Units", "systems important to safety",
                "site selection threat", "risk assessment"
            ]
        else:
            self.keywords = keywords

    def get_keywords(self):
        """
        Get the list of keywords.

        Returns:
            list: A list of keywords.
        """
        return self.keywords

    def add_keywords(self, new_keywords):
        """
        Add new keywords to the existing list.

        Args:
            new_keywords (list): A list of new keywords to add.
        """
        if isinstance(new_keywords, list):
            self.keywords.extend(new_keywords)
        else:
            raise TypeError("New keywords must be provided as a list.")

    def remove_keywords(self, keywords_to_remove):
        """
        Remove specific keywords from the list.

        Args:
            keywords_to_remove (list): A list of keywords to remove.
        """
        if isinstance(keywords_to_remove, list):
            self.keywords = [keyword for keyword in self.keywords if keyword not in keywords_to_remove]
        else:
            raise TypeError("Keywords to remove must be provided as a list.")

    def update_keywords(self, keywords):
        """
        Update the entire list of keywords.

        Args:
            keywords (list): A new list of keywords to replace the old list.
        """
        if isinstance(keywords, list):
            self.keywords = keywords
        else:
            raise TypeError("Keywords must be provided as a list.")


# %%
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
            'srlimit': 100  # Limit to the top 10 results for each keyword
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


# %%
def data_collector():
    keywords_manager = KeywordsManager()
    filtered_names = ['wiki_CNSC']
    wscraping = FilteredWikiArticleFetcher(keywords_manager=keywords_manager,
                                           file_limit=36,
                                           filtered_names=filtered_names)
    wscraping.fetch_and_save_articles()

if __name__ == "__main__":
    data_collector()



