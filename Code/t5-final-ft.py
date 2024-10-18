import transformers
from datasets import load_dataset, load_metric
import nltk
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


nltk.download('punkt')

dataset_cnndailymail = load_dataset("cnn_dailymail", '3.0.0')

dataset_cnndailymail["train"] = dataset_cnndailymail["train"].shuffle().select(range(100000))
dataset_cnndailymail["validation"] = dataset_cnndailymail["validation"].shuffle().select(range(1000))
dataset_cnndailymail["test"] = dataset_cnndailymail["test"].shuffle().select(range(1000))

model_checkpoint = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

def cln_text(input_text):
    text_without_urls = re.sub(r"http[s]?\://\S+", "", input_text)
    tokenized_sentences = nltk.sent_tokenize(text_without_urls.strip())
    split_sentences = [segment for sentence in tokenized_sentences for segment in sentence.split('\n')]
    filtered_sentences = [sentence for sentence in split_sentences if len(sentence) > 0 and sentence[-1] in string.punctuation]
    cleaned_text = '\n'.join(filtered_sentences)
    return cleaned_text

def prepare_data(data_samples):
    cleaned_texts = [cln_text(text) for text in data_samples['article']]
    formatted_inputs = [prefix + text for text in cleaned_texts]
    tokenized_inputs = tokenizer(formatted_inputs, max_length=max_input_length)

    with tokenizer.as_target_tokenizer():
        tokenized_labels = tokenizer(data_samples['highlight'], max_length=max_target_length)

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs



tokenized_datasets = dataset_cnndailymail.map(prepare_data, batched=True)

batch_size = 4
model_name = 't5-base-ft'
model_dir = f"drive/MyDrive/Models/{model_name}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=300,
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    adafactor=False,
    save_total_limit=3,
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)
data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")


def evaluate_predictions(eval_data):
    preds, refs = eval_data
    preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)

    refs = np.where(refs != -100, refs, tokenizer.pad_token_id)
    refs_decoded = tokenizer.batch_decode(refs, skip_special_tokens=True)

    preds_decoded = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds_decoded]
    refs_decoded = ["\n".join(nltk.sent_tokenize(ref.strip())) for ref in refs_decoded]

    metrics = metric.compute(predictions=preds_decoded, references=refs_decoded, use_stemmer=True)
    metrics = {metric: value.mid.fmeasure * 100 for metric, value in metrics.items()}

    gen_lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    metrics['gen_len'] = np.mean(gen_lengths)

    return {k: round(v, 4) for k, v in metrics.items()}


def model_init():
  return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=evaluate_predictions,
)

trainer.train()

from huggingface_hub import notebook_login

notebook_login()

model_name = "t5-base-ft/checkpoint-10700"
model_dir = f"drive/MyDrive/Models/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=False)
pt_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

model_name = 't5-ft'

tokenizer.push_to_hub(model_name)
pt_model.push_to_hub(model_name, safe_serialization=False)