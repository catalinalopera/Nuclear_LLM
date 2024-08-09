import os
import json
from generator import QuestionGenerator, print_qa

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
