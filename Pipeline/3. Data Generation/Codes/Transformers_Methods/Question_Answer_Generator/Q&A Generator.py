# %% [markdown]
# # Web_Scraping

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda'), "Not using CUDA. Set: Runtime > Change runtime type > Hardware Accelerator: GPU"

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
    wscraping = FilteredWikiArticleFetcher(keywords_manager=KeywordsManager(),file_limit=36,filtered_names = ['wiki_CNSC'] )
    wscraping.fetch_and_save_articles()  
if __name__ == "__main__":
    data_collector()

# %%
import zipfile
import os
class ZipExtractor:
    def __init__(self, zip_path, extract_to):
        self.zip_path = zip_path
        self.extract_to = extract_to

    def extract(self):
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)
        
        # Extract the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)

        print(f"Extracted {self.zip_path} to {self.extract_to}")

# %%
def zip_extractor():
    extractor = ZipExtractor(zip_path = 'filtered_articles.zip', extract_to = 'Articles')
    extractor.extract()
if __name__ == "__main__":
    zip_extractor()  
def list_files_in_directory(directory):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return []
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files
def print_files():
    directory = 'Articles'
    files = list_files_in_directory(directory)
    if files:
        print(f"Files in '{directory}' directory:")
        for file in files:
            print(file)
    else:
        print("No files found.")
if __name__ == "__main__":
    print_files()

# %% [markdown]
# # Q&A Generator 

# %%
import en_core_web_sm
import json
import numpy as np
import random
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from typing import Any, List, Mapping, Tuple


class QuestionGenerator:
    """A transformer-based NLP system for generating reading comprehension-style questions from
    texts. It can generate full sentence questions, multiple choice questions, or a mix of the
    two styles.

    To filter out low quality questions, questions are assigned a score and ranked once they have
    been generated. Only the top k questions will be returned. This behaviour can be turned off
    by setting use_evaluator=False.
    """

    def __init__(self) -> None:

        QG_PRETRAINED = "iarfmoose/t5-base-question-generator"
        self.ANSWER_TOKEN = ""
        self.CONTEXT_TOKEN = ""
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=False)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

        self.qa_evaluator = QAEvaluator()

    def generate(
        self,
        article: str,
        use_evaluator: bool = True,
        num_questions: bool = None,
        answer_style: str = "all"
    ) -> List:
        """Takes an article and generates a set of question and answer pairs. If use_evaluator
        is True then QA pairs will be ranked and filtered based on their quality. answer_style
        should selected from ["all", "sentences", "multiple_choice"].
        """

        print("Generating questions...\n")

        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:
            print("Evaluating QA pairs...\n")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(
                generated_questions, qg_answers
            )
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)

            if num_questions:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores, num_questions
                )
            else:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores
                )

        else:
            print("Skipping evaluation step.\n")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)

        return qa_list

    def generate_qg_inputs(self, text: str, answer_style: str) -> Tuple[List[str], List[str]]:
        """Given a text, returns a list of model inputs and a list of corresponding answers.
        Model inputs take the form "answer_token  context_token " where
        the answer is a string extracted from the text, and the context is the wider text surrounding
        the context.
        """

        VALID_ANSWER_STYLES = ["all", "sentences", "multiple_choice"]

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(
                "Invalid answer style {}. Please choose from {}".format(
                    answer_style, VALID_ANSWER_STYLES
                )
            )

        inputs = []
        answers = []

        if answer_style == "sentences" or answer_style == "all":
            segments = self._split_into_segments(text)

            for segment in segments:
                sentences = self._split_text(segment)
                prepped_inputs, prepped_answers = self._prepare_qg_inputs(
                    sentences, segment
                )
                inputs.extend(prepped_inputs)
                answers.extend(prepped_answers)

        if answer_style == "multiple_choice" or answer_style == "all":
            sentences = self._split_text(text)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(
                sentences
            )
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs: List) -> List[str]:
        """Given a list of concatenated answers and contexts, with the form:
        "answer_token  context_token ", generates a list of
        questions.
        """
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text: str) -> List[str]:
        """Splits the text into sentences, and attempts to split or truncate long sentences."""
        MAX_SENTENCE_LEN = 128
        sentences = re.findall(".*?[.!\?]", text)
        cut_sentences = []

        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split("[,;:)]", sentence))

        # remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _split_into_segments(self, text: str) -> List[str]:
        """Splits a long text into segments short enough to be input into the transformer network.
        Segments are used as context for question generation.
        """
        MAX_TOKENS = 490
        paragraphs = text.split("\n")
        tokenized_paragraphs = [
            self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0
        ]
        segments = []

        while len(tokenized_paragraphs) > 0:
            segment = []

            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

    def _prepare_qg_inputs(
        self,
        sentences: List[str],
        text: str
    ) -> Tuple[List[str], List[str]]:
        """Uses sentences as answers and the text as context. Returns a tuple of (model inputs, answers).
        Model inputs are "answer_token  context_token "
        """
        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = f"{self.ANSWER_TOKEN} {sentence} {self.CONTEXT_TOKEN} {text}"
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    def _prepare_qg_inputs_MC(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """Performs NER on the text, and uses extracted entities are candidate answers for multiple-choice
        questions. Sentences are used as context, and entities as answers. Returns a tuple of (model inputs, answers).
        Model inputs are "answer_token  context_token "
        """
        spacy_nlp = en_core_web_sm.load()
        docs = list(spacy_nlp.pipe(sentences, disable=["parser"]))
        inputs_from_text = []
        answers_from_text = []

        for doc, sentence in zip(docs, sentences):
            entities = doc.ents
            if entities:

                for entity in entities:
                    qg_input = f"{self.ANSWER_TOKEN} {entity} {self.CONTEXT_TOKEN} {sentence}"
                    answers = self._get_MC_answers(entity, docs)
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(answers)

        return inputs_from_text, answers_from_text

    def _get_MC_answers(self, correct_answer: Any, docs: Any) -> List[Mapping[str, Any]]:
        """Finds a set of alternative answers for a multiple-choice question. Will attempt to find
        alternatives of the same entity type as correct_answer if possible.
        """
        entities = []

        for doc in docs:
            entities.extend([{"text": e.text, "label_": e.label_}
                            for e in doc.ents])

        # remove duplicate elements
        entities_json = [json.dumps(kv) for kv in entities]
        pool = set(entities_json)
        num_choices = (
            min(4, len(pool)) - 1
        )  # -1 because we already have the correct answer

        # add the correct answer
        final_choices = []
        correct_label = correct_answer.label_
        final_choices.append({"answer": correct_answer.text, "correct": True})
        pool.remove(
            json.dumps({"text": correct_answer.text,
                       "label_": correct_answer.label_})
        )

        # find answers with the same NER label
        matches = [e for e in pool if correct_label in e]

        # if we don't have enough then add some other random answers
        if len(matches) < num_choices:
            choices = matches
            pool = pool.difference(set(choices))
            choices.extend(random.sample(pool, num_choices - len(choices)))
        else:
            choices = random.sample(matches, num_choices)

        choices = [json.loads(s) for s in choices]

        for choice in choices:
            final_choices.append({"answer": choice["text"], "correct": False})

        random.shuffle(final_choices)
        return final_choices

    @torch.no_grad()
    def _generate_question(self, qg_input: str) -> str:
        """Takes qg_input which is the concatenated answer and context, and uses it to generate
        a question sentence. The generated question is decoded and then returned.
        """
        encoded_input = self._encode_qg_input(qg_input)
        output = self.qg_model.generate(input_ids=encoded_input["input_ids"])
        question = self.qg_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return question

    def _encode_qg_input(self, qg_input: str) -> torch.tensor:
        """Tokenizes a string and returns a tensor of input ids corresponding to indices of tokens in
        the vocab.
        """
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_ranked_qa_pairs(
        self, generated_questions: List[str], qg_answers: List[str], scores, num_questions: int = 10
    ) -> List[Mapping[str, str]]:
        """Ranks generated questions according to scores, and returns the top num_questions examples.
        """
        if num_questions > len(scores):
            num_questions = len(scores)
            print((
                f"\nWas only able to generate {num_questions} questions.",
                "For more questions, please input a longer text.")
            )

        qa_list = []

        for i in range(num_questions):
            index = scores[i]
            qa = {
                "question": generated_questions[index].split("?")[0] + "?",
                "answer": qg_answers[index]
            }
            qa_list.append(qa)

        return qa_list

    def _get_all_qa_pairs(self, generated_questions: List[str], qg_answers: List[str]):
        """Formats question and answer pairs without ranking or filtering."""
        qa_list = []

        for question, answer in zip(generated_questions, qg_answers):
            qa = {
                "question": question.split("?")[0] + "?",
                "answer": answer
            }
            qa_list.append(qa)

        return qa_list


class QAEvaluator:
    """Wrapper for a transformer model which evaluates the quality of question-answer pairs.
    Given a QA pair, the model will generate a score. Scores can be used to rank and filter
    QA pairs.
    """

    def __init__(self) -> None:

        QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(
            QAE_PRETRAINED
        )
        self.qae_model.to(self.device)
        self.qae_model.eval()

    def encode_qa_pairs(self, questions: List[str], answers: List[str]) -> List[torch.tensor]:
        """Takes a list of questions and a list of answers and encodes them as a list of tensors."""
        encoded_pairs = []

        for question, answer in zip(questions, answers):
            encoded_qa = self._encode_qa(question, answer)
            encoded_pairs.append(encoded_qa.to(self.device))

        return encoded_pairs

    def get_scores(self, encoded_qa_pairs: List[torch.tensor]) -> List[float]:
        """Generates scores for a list of encoded QA pairs."""
        scores = {}

        for i in range(len(encoded_qa_pairs)):
            scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [
            k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _encode_qa(self, question: str, answer: str) -> torch.tensor:
        """Concatenates a question and answer, and then tokenizes them. Returns a tensor of
        input ids corresponding to indices in the vocab.
        """
        if type(answer) is list:
            for a in answer:
                if a["correct"]:
                    correct_answer = a["answer"]
        else:
            correct_answer = answer

        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def _evaluate_qa(self, encoded_qa_pair: torch.tensor) -> float:
        """Takes an encoded QA pair and returns a score."""
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]


def print_qa(qa_list: List[Mapping[str, str]], show_answers: bool = True) -> None:
    """Formats and prints a list of generated questions and answers."""

    for i in range(len(qa_list)):
        # wider space for 2 digit q nums
        space = " " * int(np.where(i < 9, 3, 4))

        print(f"{i + 1}) Q: {qa_list[i]['question']}")

        answer = qa_list[i]["answer"]

        # print a list of multiple choice answers
        if type(answer) is list:

            if show_answers:
                print(
                    f"{space}A: 1. {answer[0]['answer']} "
                    f"{np.where(answer[0]['correct'], '(correct)', '')}"
                )
                for j in range(1, len(answer)):
                    print(
                        f"{space + '   '}{j + 1}. {answer[j]['answer']} "
                        f"{np.where(answer[j]['correct']==True,'(correct)', '')}"
                    )

            else:
                print(f"{space}A: 1. {answer[0]['answer']}")
                for j in range(1, len(answer)):
                    print(f"{space + '   '}{j + 1}. {answer[j]['answer']}")

            print("")

        # print full sentence answers
        else:
            if show_answers:
                print(f"{space}A: {answer}\n")

# %%
class QuestionAnswerGenerator:
    def __init__(self, articles_folder, num_questions, answer_style='all'):
        self.articles_folder = articles_folder
        self.num_questions = num_questions
        self.answer_style = answer_style
        self.qg = QuestionGenerator()
        
    def generate_questions(self):
        """
        Generate questions from the contents of articles and split into training and testing datasets.
        """
        try:
            # List all files in the specified folder
            article_files = os.listdir(self.articles_folder)
            article_contents = []

            # Read contents of each file
            for file_name in article_files:
                file_path = os.path.join(self.articles_folder, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        article_contents.append(file.read())
                else:
                    print(f"Skipped non-file: {file_path}")

            if not article_contents:
                print("No valid article contents found.")
                return

            # Combine all article contents into a single string
            combined_article = "\n".join(article_contents)

            # Generate questions
            qa_list = self.qg.generate(combined_article, num_questions=self.num_questions, answer_style=self.answer_style)

            # Print the generated questions and answers
            print_qa(qa_list)

            # Calculate the number of question-answer pairs
            num_qa_pairs = len(qa_list)
            print(f"Total number of question-answer pairs generated: {num_qa_pairs}")

            # Format the data for JSON output
            output_data = [{"question": q['question'], "answer": q['answer']} for q in qa_list]

            # Calculate split index
            split_index = int(num_qa_pairs * 0.8)
            
            # Split data into training and testing sets
            train_data = output_data[:split_index]
            test_data = output_data[split_index:]

            # Save the training and testing data to JSON files
            with open('train.json', 'w') as f:
                json.dump(train_data, f, indent=4)
            print("Training data saved to train.json")

            with open('test.json', 'w') as f:
                json.dump(test_data, f, indent=4)
            print("Testing data saved to test.json")

        except Exception as e:
            print(f"An error occurred: {e}")

# %%
def main():  
    qag = QuestionAnswerGenerator(articles_folder = "Articles" , num_questions = 1000, answer_style = 'all')
    qag.generate_questions()
if __name__ == "__main__":
    main()

# %%
# Zarina
## Generation QA pairs for LLM fine-tuning using OpenAI API

### Overview
This script automates the extraction of text from PDF files, splits the text into manageable chunks, generates question-answer (QA) pairs using OpenAI language model, and saves the results to a JSON file. The JSON format is specifically structured for fine-tuning large language models (LLMs) with the generated QA pairs. The script utilizes the OpenAI API, which requires a paid API key for access.

### Configuration
The script begins by setting up necessary configurations, including the source directory for the PDF files, parameters for text chunking, and the OpenAI model settings. It also sets the OpenAI API key required for accessing the language model. This API key must be a valid, paid key to use the OpenAI services.

### Functions
*Extract Text from PDFs*
This function reads all PDF files in the specified directory and extracts their text content. It handles errors, ensuring that the script continues to run even if some files cannot be processed. The extracted text from each PDF is stored in a list.

*Chunk Text*
This function splits the extracted text into smaller chunks with a specified overlap. The chunking process makes it easier for the language model to process the text, as smaller chunks are more manageable. The function takes in parameters for chunk size and overlap, allowing flexibility in how the text is split.

*Generate QA Pairs*
This function generates QA pairs from the text chunks using the OpenAI language model. It runs the model on each chunk of text and parses the generated output to extract questions and answers. The function also limits the number of QA pairs generated per chunk to the specified maximum, ensuring that the output is not overly verbose and remains relevant. Since this function relies on the OpenAI API, it requires a paid API key to function.

*Save QA Pairs to JSON*
This function cleans up the QA pairs by removing any special tags (e.g., <question>, </question>, <answer>, </answer>) and saves the cleaned pairs to a JSON file. The JSON format is required for fine-tuning LLMs with the generated QA pairs. Each QA pair is saved as a dictionary with "prompt" and "response" keys.

*Main Execution*
The main execution block orchestrates the entire process. It first extracts text from all PDF files in the specified directory. Then, it splits the extracted text into chunks. For each chunk, the script generates QA pairs and collects them. Finally, it saves all generated QA pairs to a JSON file. Throughout the process, the script prints debugging information to the console, including the number of files processed, chunks created, QA pairs generated, and the final count of QA pairs saved.

### Summary of Key Features
Text Extraction: The script extracts text from PDF files in the specified directory, handling errors to ensure robustness.

Text Chunking: It splits the extracted text into smaller, manageable chunks based on specified chunk size and overlap parameters.

QA Pair Generation: The script generates QA pairs using the OpenAI language model, which requires a paid API key. It ensures relevance by limiting the number of QA pairs per chunk.

JSON Saving: Generated QA pairs are cleaned and saved in a JSON format suitable for fine-tuning large language models (LLMs).

Error Handling: The script includes basic error handling to manage issues with PDF file processing, ensuring continuity of the overall process.

# %%
!pip install openai langchain pandas tqdm PyMuPDF
!pip install -U langchain-community
!pip install openpyxl

# %%
# Importing necessary libraries
import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

# Load configuration
SOURCE_DIRECTORY = '/Users/zarinadossayeva/Desktop/WIL_LLM/CNSC_QA_pairs_JSON/CNSC_docs_1_10'
chunk_size = 1000
chunk_overlap = 100
model = "gpt-3.5-turbo-0125"
temperature = 0
max_tokens = None
max_qa_per_chunk = 5  # Limiting the number of QA pairs per chunk

# Set OpenAI API key
api_key = 'open-api-key'
os.environ['OPENAI_API_KEY'] = api_key

# Verify API key is set
print(f"Using OpenAI API key: {os.environ.get('OPENAI_API_KEY')}")  # Debugging line

# Define a function to read all PDFs and extract text
def extract_text_from_pdfs(directory):
    """Extracts text from all PDF files in the specified directory.

    Args:
        directory (str): The directory containing PDF files.

    Returns:
        list: A list of strings, each containing the extracted text from a PDF file.
    """
    text_data = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")  # Debugging line

    if not pdf_files:
        print("No PDF files found in the specified directory.")  # Debugging line

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file}")  # Debugging line
        try:
            loader = PyMuPDFLoader(file_path=pdf_file)
            documents = loader.load()  # Expecting a list of documents
            for document in documents:
                text = document.page_content
                text_data.append(text)
                print(f"Extracted text from {pdf_file} (length {len(text)}): {text[:500]}...")  # Debugging line
        except Exception as e:
            print(f"Error processing file {pdf_file}: {e}")  # Debugging line

    return text_data

# Convert documents to chunks
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Splits the text into smaller chunks with a specified overlap.

    Args:
        text (str): The text to be split.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 100.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    print(f"Chunked text into {len(chunks)} chunks (total length {len(text)}):")  # Debugging line
    if chunks:
        print(f"First chunk: {chunks[0]}")  # Debugging line
    return chunks

# Prompt to use OpenAI API as QA generator
qa_prompt_template = """
You are an intelligent and supportive assistant.
Your task is to create question-answer pairs from the given context.
Question type: Context based/Yes-No/ short question answer/ long question answer.
Just use the information in the context to write question and answer.
Please don't make up anything outside the given context.

Text: {context}

Generate as many question-answer pairs as possible. Always use tags to enclose question answers as shown in below examples.

<question>What is SMR?</question>
<answer>SMR stands for Small Modular Reactors, which are smaller, more flexible nuclear energy plants that can be deployed in various settings, including large established grids, smaller grids, remote off grid communities, and resource projects. They are designed to provide non-emitting baseload generation and can support intermittent renewable sources like wind and solar. They are also capable of producing steam for industrial purposes.</answer>
<question>What are the key objectives of the SMR project at the Darlington site in Ontario?</question>
<answer>The key objectives of the SMR project at the Darlington site in Ontario are to maintain a diverse generation supply mix to minimize carbon emissions from electricity generation in the province, to demonstrate a First-Of-A-Kind (FOAK) SMR to be ready for deployment across Canada by 2030, and to ensure economic development by securing Canadian content both for domestic and export projects from the developer in exchange for providing the opportunity to deploy their FOAK unit and be a first mover towards an SMR fleet.</answer>
... (continue as needed)
"""

# Initialize the LangChain prompt
qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context"])
llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

# Create an LLMChain
qa_chain = LLMChain(prompt=qa_prompt, llm=llm)

# Generate QA pairs for every chunk
def generate_qa_pairs(chunks, max_qa_per_chunk):
    """Generates question-answer pairs from text chunks using the language model.

    Args:
        chunks (list): The list of text chunks.
        max_qa_per_chunk (int): The maximum number of QA pairs to generate per chunk.

    Returns:
        list: A list of dictionaries, each containing a prompt and response.
    """
    qa_pairs = []
    for i, chunk in enumerate(tqdm(chunks, desc="Generating QA pairs")):
        try:
            print(f"Processing chunk {i + 1}/{len(chunks)} (length {len(chunk)}): {chunk[:500]}...")  # Debugging line
            response = qa_chain.run({"context": chunk})
            print(f"Generated QA pairs for chunk {i + 1}/{len(chunks)}: {response}")  # Debugging line
            
            # Parse the QA pairs from the response
            qa_list = response.split('<question>')
            for qa in qa_list[1:max_qa_per_chunk+1]:
                if '<answer>' in qa:
                    question = "<question>" + qa.split('</question>')[0] + "</question>"
                    answer = "<answer>" + qa.split('<answer>')[1].split('</answer>')[0] + "</answer>"
                    qa_pairs.append({"prompt": question, "response": answer})
                else:
                    qa_pairs.append({"prompt": qa.strip(), "response": ""})
            
        except Exception as e:
            print(f"Error generating QA for chunk {i + 1}/{len(chunks)}: {e}")
    return qa_pairs

# Save questions and answers to a JSON file
def save_questions_to_json(qa_pairs, output_file="CNSC_QA_pairs_1_10.json"):
    """Saves the generated QA pairs to a JSON file.

    Args:
        qa_pairs (list): The list of QA pairs to save.
        output_file (str, optional): The name of the output JSON file. Defaults to "questions.json".
    """
    cleaned_qa_pairs = []
    for qa in qa_pairs:
        prompt = qa['prompt'].replace("<question>", "").replace("</question>", "").strip()
        response = qa['response'].replace("<answer>", "").replace("</answer>", "").strip()
        cleaned_qa_pairs.append({"prompt": prompt, "response": response})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_qa_pairs, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(cleaned_qa_pairs)} QA pairs to {output_file}")

# Main execution
if __name__ == "__main__":
    """Main execution function to run the entire pipeline: 
    extract text, chunk text, generate QA pairs, and save to JSON.
    """
    text_data = extract_text_from_pdfs(SOURCE_DIRECTORY)
    all_qa_pairs = []
    
    for text in text_data:
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        qa_pairs = generate_qa_pairs(chunks, max_qa_per_chunk)
        all_qa_pairs.extend(qa_pairs)
    
    print(f"Total QA pairs generated: {len(all_qa_pairs)}")  # Printing total QA pairs generated
    save_questions_to_json(all_qa_pairs)

# %%
import json

def truncate_json(input_file, output_file, max_pairs):
    # Read JSON file with utf-8 encoding
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Limit the number of pairs
    if len(data) > max_pairs:
        data = data[:max_pairs]

    # Write the results to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# File paths for training and testing data
train_file = 'train.json'
test_file = 'test.json'
train_output_file = 'train_truncated.json'
test_output_file = 'test_truncated.json'

# Maximum number of pairs for training and testing
max_train_pairs = 350
max_test_pairs = 150

# Process training and testing data
truncate_json(train_file, train_output_file, max_train_pairs)
truncate_json(test_file, test_output_file, max_test_pairs)

print(f"Created {train_output_file} with {max_train_pairs} pairs from {train_file}.")
print(f"Created {test_output_file} with {max_test_pairs} pairs from {test_file}.")


# %%
import json
import os

def count_pairs(input_file):
    """Count the number of question-answer pairs in a JSON file."""
    if not os.path.isfile(input_file):
        print(f"Error: {input_file} not found.")
        return 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}")
        return 0
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return 0

def transform_keys(data):
    """Transform 'prompt' to 'question' and 'response' to 'answer'."""
    transformed_data = []
    for item in data:
        if 'prompt' in item and 'response' in item:
            transformed_item = {
                'question': item['prompt'],
                'answer': item['response']
            }
            transformed_data.append(transformed_item)
        else:
            print(f"Skipping item due to missing keys: {item}")
    return transformed_data

def split_and_save_data(input_file, train_output_file, test_output_file, max_train_pairs, max_test_pairs):
    """Split the data into training and testing datasets, transform keys, and save to new files."""
    if not os.path.isfile(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Transform the keys
        data = transform_keys(data)
        
        # Determine how many pairs to include in each file
        total_pairs = len(data)
        print(f"Total pairs in {input_file}: {total_pairs}")
        
        # Truncate data based on maximum pairs
        train_data = data[:max_train_pairs]
        test_data = data[max_train_pairs:max_train_pairs + max_test_pairs]
        
        # Write the results to new files
        with open(train_output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        
        with open(test_output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        
        print(f"Created {train_output_file} with up to {max_train_pairs} pairs.")
        print(f"Created {test_output_file} with up to {max_test_pairs} pairs.")
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# File paths
input_file = 'CNSC_QA_pairs_1_10.json'
train_output_file = 'train.json'
test_output_file = 'test.json'

# Maximum number of pairs for training and testing
max_train_pairs = 850
max_test_pairs = 150

# Count the pairs in the input file
total_pairs = count_pairs(input_file)
print(f"Total number of pairs in {input_file}: {total_pairs}")

# Split, transform, and save the data
split_and_save_data(input_file, train_output_file, test_output_file, max_train_pairs, max_test_pairs)



