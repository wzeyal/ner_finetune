import math
import datasets 
import numpy as np 
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification 
from transformers import TrainingArguments, Trainer 
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score



writer = SummaryWriter()

def log_to_tensorboard(metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)
        
def flatten_tags(tags):
    return [tag[2:] if tag.startswith(("B-", "I-")) else tag for tag in tags]

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
    # fig, _ = plt.subplots()
    fig = plt.figure(figsize=(10, 10))
    # im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # fig.colorbar(im)
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion Matrix')
    
    # Convert the plot to a torch.Tensor
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    tensor_image = torch.from_numpy(image_from_plot).permute(2, 0, 1).float() / 255.0
    plt.close()

    return tensor_image


def compute_metrics(eval_preds, bios_label_list, epoch, plot_confusion_matrix=False): 
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
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
    
    if plot_confusion_matrix:
        flatten_labels = flatten_bio_tags_with_index(bios_label_list)
    
        conf_matrix = calculate_confusion_matrix(predictions, true_labels, flatten_labels).astype('float') 
        
        # norm_conf = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        normalized_conf_matrix = conf_matrix / (conf_matrix.sum(axis=0, keepdims=True) + 1e-8)
        
        tensor_image = confusion_matrix_to_tensor(conf_matrix, flatten_labels)
        writer.add_image("Confusion Matrix", tensor_image, epoch)
        
        tensor_image = confusion_matrix_to_tensor(normalized_conf_matrix, flatten_labels, ".0%")
        writer.add_image("Normalized Matrix", tensor_image, epoch)
        
        # results = metric.compute(predictions=predictions, references=true_labels) 
        # report = classification_report(predictions, true_labels, target_names=bios_label_list, output_dict=True)
    
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
    } 
    
    log_to_tensorboard(result, epoch)
    # log_to_tensorboard(results['LOC'], epoch)
    return result


def main():
    conll2003 = datasets.load_dataset("conll2003") 
    label_list = conll2003["train"].features["ner_tags"].feature.names
    
    id2label = dict(zip(range(len(label_list)), label_list))
    label2id = {v: k for k, v in id2label.items()}
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") 
    tokenized_datasets = conll2003.map(lambda example: tokenize_and_align_labels(example, tokenizer), batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
        id2label=id2label, label2id=label2id)

    print(torch.cuda.is_available())
    
    args = TrainingArguments( 
        "test-ner",
        evaluation_strategy = "epoch", 
        save_strategy = "epoch",
        learning_rate=2e-5, 
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4, 
        per_device_eval_batch_size=4, 
        num_train_epochs=15, 
        weight_decay=0.01,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    ) 
    
    data_collator = DataCollatorForTokenClassification(tokenizer) 
    
    # train_dataset = tokenized_datasets["train"].select(range(50))
    # eval_dataset = tokenized_datasets["train"].select(range(1))
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    
    trainer = Trainer( 
        model, 
        args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=data_collator, 
        tokenizer=tokenizer, 
        compute_metrics=lambda x: compute_metrics(x, bios_label_list=label_list, epoch=trainer.state.epoch),
    )    
    
    trainer.train()
    
    print(trainer.state)
    
    # Evaluate the model
    predictions, labels, _ = trainer.predict(eval_dataset)

    compute_metrics((predictions, labels), label_list, epoch=None, plot_confusion_matrix=True)
    

if __name__ == "__main__":
    main()