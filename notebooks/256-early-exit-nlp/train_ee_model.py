import argparse
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from ee_model.configuration_ee import EarlyExitConfig
from ee_model.modeling_ee import EarlyExitModelForQuestionAnswering


def preprocess_squad2_dataset(tokenizer=None, num_samples=-1):
    """Function to preprocess and tokenize squad dataset before training model"""
    data_split = "train" if num_samples == -1 else f"train[:{num_samples}]"
    squadv2 = load_dataset("squad_v2", split=data_split)

    # Remove questions with blank answers from the dataset
    empty_answer_indices = []
    for i, example in enumerate(squadv2):
        if len(example["answers"]["text"]) == 0 or len(example["answers"]["answer_start"]) == 0:
            empty_answer_indices.append(i)
    squadv2 = squadv2.select(index for index in range(len(squadv2)) if index not in empty_answer_indices)
                
    # Taken from HuggingFace Question-Answering tutorial
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            #print(answer)

            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return squadv2.map(preprocess_function, batched=True, remove_columns=squadv2.column_names)


class EarlyExitModelTrainer:
    """Class for training an early exit model in phases"""
    def __init__(self, model=None, save_dir=None):
        self.model = model
        self.save_dir = save_dir
        self.training_phase = -1 # -1 represents phase hasn't been set
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_training_phase(self, phase=1):
        """Setter for training phase (1 or 2)"""
        if phase == self.training_phase:
            return
        
        if phase == 1:
            ## Freeze n-1 ramp classifiers
            for params in self.model.ramp_classifiers[:-1].parameters():
                params.requires_grad = False
            self.training_phase = 1
        elif phase == 2:
            ## Freeze all parameters execept n-1 ramp classifiers
            for params in self.model.parameters():
                params.requires_grad = False    
            for params in self.model.ramp_classifiers[:-1].parameters():
                params.requires_grad = True
            self.training_phase = 2
        else:
            raise ValueError('Training phase can be 1 or 2 only.')

    def get_training_phase(self):
        """Getter for training phasse"""
        return self.training_phase

    def save_model_checkpoint(self):
        """Function for saving model checkpoints during training"""
        self.model.save_pretrained(self.save_dir)

    def train(self, dataset=None, **kwargs):
        """Function to train model"""
        # Set up training configuration
        batch_size = kwargs.get("batch_size", 32)
        lr = kwargs.get("lr", 3e-5)
        num_epochs = kwargs.get("num_epochs", 3)
        warmup_proportion = kwargs.get("warmup_proportion", 0.2)

        # Define dataloader
        loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True) if self.device == torch.device("cuda") else dict(shuffle=True, batch_size=batch_size)
        train_loader = DataLoader(dataset, **loader_args)

        # Prepare for model training
        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps = len(train_loader)*num_epochs
        num_warmup_steps = int(warmup_proportion*num_training_steps)  
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        self.model.train()

        # Training loop
        for epoch in range(num_epochs):
            avg_step_loss = 0
            num_steps = 0

            for qa_pair in train_loader:
                optimizer.zero_grad()

                input_ids = torch.transpose(torch.stack(qa_pair['input_ids']), 0, 1).to(self.device)
                attention_mask = torch.transpose(torch.stack(qa_pair['attention_mask']), 0, 1).to(self.device)
                start_positions = qa_pair['start_positions'].to(self.device)
                end_positions = qa_pair['end_positions'].to(self.device)

                output =  self.model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, training=True, training_phase=self.training_phase)
                loss = output.loss
                avg_step_loss += loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                num_steps += 1

                # Display training loss and save model checkpoint after every 500 steps
                if num_steps % 500 == 0:
                    print(f"Loss at step {num_steps}: {loss}")
                    self.save_model_checkpoint()
  
            avg_step_loss /= num_steps
            print(f"Epoch {epoch+1} Complete, Average Loss: {avg_step_loss}")

            # Save model checkpoint after each epoch
            self.save_model_checkpoint()
            print(f"Saved model checkpoint after epoch {epoch+1} in")

        self.model.eval()


def load_model_checkpoint(save_dir=None):
    """Function to load model checkpoints if required"""
    return EarlyExitModelForQuestionAnswering.from_pretrained(save_dir)


def parse_parameters():
    parser = argparse.ArgumentParser(description="""Train Early Exit Model""")
    parser.add_argument("--base_model", action="store", dest="base_model", required=False, default="bert-base-uncased", help="""--- Base HuggingFace Model for building Early Exit model. This model should have embeddings and encoder attributes ---""")
    parser.add_argument("--dataset_size", action="store", dest="dataset_size", required=False, default=-1, help="""--- Number of samples from the SQUAD v2 dataset to train the model. Defaults to -1 which uses the entire SQUAD v2 training dataset ---""")
    parser.add_argument("--batch_size", action="store", dest="batch_size", required=False, default=32, help="""--- Batch Size for finetuning model. Defaults to 32 ---""")
    parser.add_argument("--save_dir", action="store", dest="save_dir", required=False, default="./ee_model_ckpt", help="""--- Directory to save trained model ---""")
    parser.add_argument("--num_epochs", action="store", dest="num_epochs", required=False, default=3, help="""--- Number of epochs for each phase. Defaults to 3 ---""")
    parser.add_argument("--lr", action="store", dest="lr", required=False, default=3e-5, help="""--- learning rate. Defaults to 3e-5 ---""")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_parameters()

    # Initialize an Early Exit model object and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = EarlyExitModelForQuestionAnswering(EarlyExitConfig(base_model=args.base_model))

    # Prepare dataset for training
    tokenized_squad2 = preprocess_squad2_dataset(tokenizer=tokenizer, num_samples=int(args.dataset_size))

    # Initialize model trainer
    trainer = EarlyExitModelTrainer(model=model, save_dir=args.save_dir)
    
    # Finetuning Phase 1
    print("Starting Phase 1 finetuning..")
    trainer.set_training_phase(phase=1)
    trainer.train(dataset=tokenized_squad2, batch_size=args.batch_size, lr=args.lr, num_epochs=args.num_epochs)
    print("Phase 1 finetuning complete!")

    # Finetuning Phase 2
    print("Starting Phase 2 finetuning..")
    trainer.set_training_phase(phase=2)
    trainer.train(dataset=tokenized_squad2, batch_size=args.batch_size, lr=args.lr, num_epochs=args.num_epochs)
    print("Phase 2 finetuning complete!")

    # Load the most recent checkpoint
    model = load_model_checkpoint(save_dir=args.save_dir)

    ## Test the trained model on some question-context pairs from the squad dataset
    model.eval()

    print("Testing finetuned model on some question-context pairs..")
    questions = ["Which name is also used to describe the Amazon rainforest in English?", "Where do I live?"]
    contexts =  ["The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.", "My name is Clara and I live in Berkeley."]

    # Inference using pipeline class from transformers
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1) # id 0 for 1 gpu, -1 for cpu
    print(qa_pipe(question=questions[0], context=contexts[0]))
    print(qa_pipe(question=questions[1], context=contexts[1]))

    # Inference using pipeline class and non-zero entropy threshold
    model.set_entropy_threshold(2.0)
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    print(qa_pipe(question=questions[0], context=contexts[0]))
    print(qa_pipe(question=questions[1], context=contexts[1]))


if __name__ == "__main__":
    main()
