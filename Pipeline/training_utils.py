# training_utils.py
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

class ModelTrainer:
    def __init__(self, model, tokenizer, train_dataset, test_dataset, output_dir='results'):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir

    def train(self, epochs=3, per_device_train_batch_size=8, per_device_eval_batch_size=8, logging_dir='logs'):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_dir=logging_dir,
            logging_steps=10,
            save_steps=10,
            evaluation_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )

        trainer.train()

    def evaluate(self):
        # Perform evaluation
        print("Evaluating the model...")
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=self.output_dir)
        )
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)
        return eval_results
