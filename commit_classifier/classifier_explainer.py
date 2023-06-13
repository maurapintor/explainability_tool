from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
from model.tokenizer import CustomTokenizer
from explainer import CommitExplainer
import os
from data_processing import get_sample_from_url
from utils import download_gdrive


class ClassifierExplainer:

    def __init__(self, device="cpu"):

        model_name = "CAUKiel/JavaBERT-uncased"
        model_path = os.path.join(
            os.path.dirname(__file__), f"model/model.bin")
        if not os.path.exists(model_path):
            download_gdrive("1yFaMCk0gi2O494Y2w0gQK-9ReMxgdA8b", model_path)

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config)
        tokenizer = CustomTokenizer.from_pretrained(model_name)
        state_dict = torch.load(model_path, map_location=device)
        keys = list(state_dict.keys())
        for k in keys:
            state_dict[k.partition("encoder.")[-1]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.label_names = ["Not relevant", "Security relevant"]

        self.commit_explainer = CommitExplainer(
            model, tokenizer, custom_labels=self.label_names)

    def classify_explain(self, commit_url):
        sample = get_sample_from_url(commit_url)
        true_class = self.label_names[sample["label"]] \
            if "label" in sample else None
        return self.commit_explainer.explain_visualize(
            sample, commit_url=commit_url, html_filepath="aa.html",
            true_class=true_class)


if __name__ == "__main__":
    ClassifierExplainer(device="cuda").classify_explain(
        "https://github.com/jenkinsci/junit-plugin/commit/"
        "15f39fc49d9f25bca872badb48e708a8bb815ea7")
