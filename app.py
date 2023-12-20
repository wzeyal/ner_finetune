from datetime import datetime
import io
import os
import comet_ml 
import json
import math
import datasets 
import numpy as np 
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification 
from transformers import TrainingArguments, Trainer 
# from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from PIL import Image
from spacy import displacy
import shutil



# experiment = comet_ml.Experiment(
#   api_key="NgggnOHlNpfBbDkrtyEDzFRQt",
#   project_name="ner",
#   workspace="wzeyal"
# )


def flatten_bio_tags_with_index(bio_tags):
  """
  Flattens a list of Bio-NER tags to individual entity types with a running index.

  Args:
    bio_tags: A list of Bio-NER tag names.

  Returns:
    A list of entity type names with a running index appended.
  """

  flat_tags = []
  running_index = 1
  entity_type = None

  for tag in bio_tags:
    # Extract entity type from Bio-NER tag
    new_entity_type = tag.split("-")[1] if "-" in tag else "O"

    # Check for entity type change
    if new_entity_type != entity_type:
      entity_type = new_entity_type
      running_index = 1  # Reset index for new entity

    # Add entity type with index
    flat_tags.append(f"{entity_type}-{running_index}")
    running_index += 1


def flatten_bio_tags_with_index(bio_tags):
  """
  Flattens a list of Bio-NER tags to individual entity types with a running index.

  Args:
    bio_tags: A list of Bio-NER tag names.

  Returns:
    A list of flat entity type
  """

  flat_tags = []
  
  for tag in bio_tags:
    # Extract entity type from Bio-NER tag
    new_entity_type = tag.split("-")[1] if "-" in tag else "O"
    if new_entity_type not in flat_tags:
      flat_tags.append(new_entity_type)

  return flat_tags
        
def calculate_confusion_matrix(true_tags, pred_tags, fatten_labels):
    """
    Calculates the confusion matrix for a given set of true and predicted NER tags.

    Args:
        true_tags: A list of lists of true NER tags.
        pred_tags: A list of lists of predicted NER tags.
        label2id: A dictionary mapping NER labels to their IDs.
        id2label: A dictionary mapping NER IDs to their labels.

    Returns:
        A confusion matrix as a 2D numpy array.
    """
    num_labels = len(fatten_labels)
    confusion_matrix = np.zeros((num_labels, num_labels))
    for true_tag_seq, pred_tag_seq in zip(true_tags, pred_tags):
        for true_tag, pred_tag in zip(true_tag_seq, pred_tag_seq):
            true_id = fatten_labels.index(true_tag.split('-')[-1])
            pred_id = fatten_labels.index(pred_tag.split('-')[-1])
            confusion_matrix[true_id, pred_id] += 1

    # confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # experiment.log_confusion_matrix(true_tags, pred_tags, title="comet_ml_confusion_matrix")

    return confusion_matrix
        

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True): 
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Parameters:
    examples (dict): A dictionary containing the tokens and the corresponding NER tags.
                     - "tokens": list of words in a sentence.
                     - "ner_tags": list of corresponding entity tags for each word.
                     
    label_all_tokens (bool): A flag to indicate whether all tokens should have labels. 
                             If False, only the first token of a word will have a label, 
                             the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token. 
        previous_word_idx = None 
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None 
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids: 
            if word_idx is None: 
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token                 
                label_ids.append(label[word_idx]) 
            else: 
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                # mask the subword representations after the first subword
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs

def confusion_matrix_to_tensor(conf_matrix, labels, fmt=",.0f"):
    # fig, _ = plt.subplocalculate_confusion_matrixts()
    fig = plt.figure(figsize=(10, 10))
    # im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # fig.colorbar(im)
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    fig = plt.gcf()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img = Image.open(buffer)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    return img_tensor

def format_number(num): 
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "{:.1f}{}".format(num, unit)
        num /= 1000.0

    return "{:.1f}{}".format(num, 'Y')

def confusion_matrix_bar_to_figure(ner_confusion_matrix, labels, percentage_threshold=0.05):
    
    ner_confusion_matrix[:, 2] = 0
    ner_confusion_matrix[2, :] = 0
    
    ner_confusion_matrix[:, 3] = 0
    ner_confusion_matrix[3, :] = 0
    
    zero_rows = np.all(ner_confusion_matrix == 0, axis=1)
    zero_cols = np.all(ner_confusion_matrix == 0, axis=0)

    # Identify rows and columns where both the row and column are zero
    zero_rows_and_cols = zero_rows & zero_cols

    # Remove both zero rows and columns
    filtered_data = np.delete(np.delete(ner_confusion_matrix, np.where(zero_rows_and_cols), axis=0), np.where(zero_rows_and_cols), axis=1)
    filterd_labels = [label for label, condition in zip(labels, np.logical_not(zero_rows_and_cols)) if condition]
        

    
    # Calculate percentages
    total_per_actual = np.sum(filtered_data, axis=1)
    percentages = (filtered_data.T / total_per_actual).T

    # Create a horizontal stacked bar plot with percentages
    fig, ax = plt.subplots(figsize=(20, 20))
    
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    # Plot the confusion matrix
    bottom = np.zeros(len(filterd_labels))
    for i, actual_label in enumerate(filterd_labels):
        bars = ax.barh(filterd_labels, percentages[:, i], label=actual_label, left=bottom)
        bottom += percentages[:, i]
        
        # Add labels to each bar if the percentage is above 10%
        for pred_id, (bar, entity_label) in enumerate(zip(bars, filterd_labels)):
            if entity_label==filterd_labels[i]:
                bar.set_hatch("//")
            percentage = percentages[filterd_labels.index(entity_label), i]
            # nof_entities1 = int(percentage*total_per_actual[i])
            nof_entities = filtered_data[pred_id, i]
            nof_entities = format_number(nof_entities)
            if percentage > percentage_threshold:
                bar_label = f"{filterd_labels[i]}\n{percentage:.0%}\n{nof_entities}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, bar_label,
                        ha='center', va='center', color='white', fontweight='bold', fontsize=20)

    # Set labels and ticks
    ax.set_xlabel('Percentage', fontsize=20)
    ax.set_ylabel('Actual Labels', fontsize=20)

    # Set title
    ax.set_title('NER Confusion Matrix - Horizontal Stacked Bar (Percentage)')

    # Add legend
    ax.legend(fontsize=20)
    
    return plt.gcf()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img = Image.open(buffer)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    return img_tensor


def compute_metrics(eval_preds, bios_label_list, experiment : comet_ml.Experiment, epoch): 
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score, accuracy and the confusion matrix.
    """
    conf_matrix = None
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2) 
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    
    # We remove all the values where the label is -100
    predictions = [ 
        [bios_label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
    
    true_labels = [ 
      [bios_label_list[l] for (_, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
    ] 
    

    flatten_labels = flatten_bio_tags_with_index(bios_label_list)
    conf_matrix = calculate_confusion_matrix(true_labels, predictions, flatten_labels).astype('float') 
    conf_matrix_bar = confusion_matrix_bar_to_figure(conf_matrix, flatten_labels)
    report = classification_report(true_labels, predictions, output_dict=True)
    
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)

    # return report
    result =  { 
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "accuracy": accuracy,
        # "report": report,
        # 'confusion_matrix': conf_matrix,
    } 
    
    experiment.log_confusion_matrix(matrix=conf_matrix, labels=flatten_labels, epoch=epoch)
    experiment.log_figure(figure_name="Confusion Matrix (%)", figure=conf_matrix_bar, step=epoch)
    
    experiment.log_text(datetime.now().timestamp())
    
    for report_key, report_value in report.items():
        if isinstance(report_value, dict):
            for metric_key, metric_value in report_value.items():
                result[f"{report_key}_{metric_key}"] = metric_value
        else:
            result[report_key] = report_value
    
    # from pydantic.utils import deep_update

    
    # result = deep_update(result, report)
    # result['confusion_matrix'] = conf_matrix.tolist()

    return result


def main():
    
    
    output_path = "test-ner"
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    # experiment.log_text("Experiment Name", "test-ner")
    conll2003 = datasets.load_dataset("conll2003") 
    label_list = conll2003["train"].features["ner_tags"].feature.names
    
    id2label = dict(zip(range(len(label_list)), label_list))
    label2id = {v: k for k, v in id2label.items()}
    
    # You need to specify the algorithm and hyperparameters to use:
    config = {
        # Pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters:
        "parameters": {
            "train_batch_size": {"type": "discrete", "values": [4, 8]},
            "gradient_accumulation_steps": {"type": "discrete", "values": [4, 8, 16]},
            "weight_decay": {"type": "discrete", "values": [0, 0.0001]},
        },

        # Declare what to optimize, and how:
        "spec": {
            "maxCombo": 12,
            "metric": "eval/f1",
            "objective": "maximize",
        },
    }
    
    opt = comet_ml.Optimizer(config)
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") 
    tokenized_datasets = conll2003.map(lambda example: tokenize_and_align_labels(example, tokenizer), batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
        id2label=id2label, label2id=label2id)
    
    timestamp = datetime.now().timestamp()

    
    for index, exp in enumerate(opt.get_experiments(
        project_name="ner")):
        
        exp.add_tag(timestamp)
        exp.set_name(f"{index}")

        params = exp.params
        
        args = TrainingArguments( 
            output_path,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5, 
            per_device_train_batch_size=params["train_batch_size"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            per_device_eval_batch_size=8, 
            num_train_epochs=2, 
            weight_decay=params["weight_decay"],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=3,
            report_to="tensorboard",
        ) 
        
        data_collator = DataCollatorForTokenClassification(tokenizer) 
        
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        
        train_dataset = tokenized_datasets["train"].select(range(32))
        eval_dataset = tokenized_datasets["train"].select(range(32))
                
        trainer = Trainer( 
            model, 
            args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            data_collator=data_collator, 
            tokenizer=tokenizer, 
            compute_metrics=lambda x: compute_metrics(
                x, bios_label_list=label_list, experiment=exp, epoch=trainer.state.epoch
            ),
        )    
        
        trainer.train()
        exp.end()
    
    return
    
    # best_metric = trainer.state.best_metric
    
    # evalulate_metric = trainer.evaluate()
    
    # if 'eval_confusion_matrix' in evalulate_metric.keys():
    #     conf_matrix = evalulate_metric['eval_confusion_matrix']
    #     conf_matrix = np.array(conf_matrix)
    #     normalized_conf_matrix = conf_matrix / (conf_matrix.sum(axis=0, keepdims=True) + 1e-8)
    #     print(normalized_conf_matrix)
        
    #     flatten_labels = flatten_bio_tags_with_index(label_list)
        
    #     tensor_image = confusion_matrix_to_tensor(conf_matrix, flatten_labels)
    #     writer.add_image("Confusion Matrix", tensor_image)
        
    #     tensor_image = confusion_matrix_to_tensor(normalized_conf_matrix, flatten_labels, ".0%")
    #     writer.add_image("Normalized Matrix", tensor_image)
        
    #     tensor_image = confusion_matrix_bar_to_tensor(conf_matrix, flatten_labels)
    #     writer.add_image("Confusion Bar Matrix", tensor_image)
        
    #     experiment.log_confusion_matrix(matrix=conf_matrix, labels=flatten_labels)
    #     experiment.log_confusion_matrix(matrix=normalized_conf_matrix, labels=flatten_labels)
        
    from transformers import pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")

    # df = pd.DataFrame({'sentence': ["This is a test sentence", 'This is another test sentence', ]})
    
    df = eval_dataset.to_pandas()
    df['sentence'] = df['tokens'].apply(" ".join)

    df['ner_results'] = df['sentence'].apply(lambda text: ner_pipeline(text))
    df['ents'] = df['ner_results'].apply(
        lambda ner_results: [
            {
                'start': ent['start'], 
                'end': ent['end'],
                'label': ent['entity_group']
            }
            for ent in ner_results
        ]
    )
    df['spacy_format'] = df.apply(lambda row:
        {
            'text': row['sentence'],
            'ents': row['ents'],
        }
        , axis=1
    )
    
    df['spacy_html'] = df['spacy_format'].apply(lambda spacy_format: displacy.render(spacy_format, style="ent", manual=True, page=True))
    
    # sample_df = df.iloc[0]
    # spacy_format = {
    #     'text': sample_df['sentence'],
    #     'ents': sample_df['ents'],
    # }
    
    for index, row in df.iterrows():
        rtl_html = row['spacy_html'].replace('ltr', 'ltr')
        experiment.log_html(f"<h2>Sample {index}</h2>")
        experiment.log_html(rtl_html)
        experiment.log_html("<hr>")

    
    
    
    # html = displacy.render(spacy_format, style="ent", manual=True)
    # print(html)
    pass


if __name__ == "__main__":
    main()