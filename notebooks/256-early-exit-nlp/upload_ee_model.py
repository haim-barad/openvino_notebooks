import argparse
from transformers import AutoTokenizer

from ee_model.configuration_ee import EarlyExitConfig
from ee_model.modeling_ee import EarlyExitModelForQuestionAnswering


def parse_parameters():
    parser = argparse.ArgumentParser(description="""Upload Early Exit Model to HuggingFace Hub""")
    parser.add_argument("--save_dir", action="store", dest="save_dir", required=False, default="./ee_model_ckpt", help="""--- Directory where finetuned model is stored ---""")
    parser.add_argument("--model_repo", action="store", dest="model_repo", required=False, default="ee-qa-model", help="""--- Model repo to push finetuned model to ---""")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_parameters()

    # register custom config and model classes with huggingface autoclasses
    EarlyExitConfig.register_for_auto_class()
    EarlyExitModelForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")

    # Initialize model and tokenizer from saved checkpoints
    model = EarlyExitModelForQuestionAnswering.from_pretrained(args.save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model)

    # Push model and tokenizer to HuggingFace Hub
    tokenizer.push_to_hub("ee-qa-model", private=True)
    print("Successfully pushed tokenizer to model repo.")
    model.push_to_hub("ee-qa-model", private=True)
    print("Successfully pushed model to model repo.")


if __name__ == "__main__":
    main()
