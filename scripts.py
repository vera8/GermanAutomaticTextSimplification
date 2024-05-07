import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from transformers.optimization import Adafactor
from datasets import Dataset, load_dataset
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from easse.bertscore import corpus_bertscore


def load_model(model_id, tokenizer_id=None, hf_token=None):
    """Loads a model and a tokenizer for a given id"""
    print(model_id)
    if tokenizer_id is None:
        tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, legacy=False, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=hf_token)
    return model, tokenizer


def load_peft_model(peft_model_id, base_model_id, tokenizer_id=None, hf_token=None, resize_embedding=False):
    """Loads a peft model based on a peft model id and a base model id, as well as a tokenizer"""
    if tokenizer_id is None:
        tokenizer_id = base_model_id
    print(base_model_id)
    print(peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, legacy=False, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, device_map="auto", token=hf_token)
    if resize_embedding:
        print(f"Resizing embedding layer to {tokenizer.vocab_size}...")
        model.resize_token_embeddings(tokenizer.vocab_size)
    model.load_adapter(peft_model_id, token=hf_token)
    return model, tokenizer


def trim_mt5_model(model, tokenizer):
    """Trims the vocabulary of an mT5-Model. Needs a keps_ids.json to work (can be generated via
    filter_tokenizer_vocabulary.ipynb, but is already included in the project directory)"""
    print("Trimming model vocabulary...")
    input_embed_weights_old = model.get_input_embeddings().weight.clone().detach()
    output_embed_weights_old = model.get_output_embeddings().weight.clone().detach()
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    with open("kept_ids.json", "r") as f:
        kept_ids = json.load(f)
    
    def shrink_embedding(emb_new, emb_old_weights):
        i = 0
        for idx in kept_ids:
            weight = emb_old_weights[idx]
            with torch.no_grad():
                emb_new.weight[i] = weight
            i += 1
    
    input_embed_new = nn.Embedding(tokenizer.vocab_size, model.model_dim)
    shrink_embedding(input_embed_new, input_embed_weights_old)
    model.set_input_embeddings(new_embeddings=input_embed_new)
    
    output_embed_new = nn.Linear(model.model_dim, tokenizer.vocab_size, bias = False)
    shrink_embedding(output_embed_new, output_embed_weights_old)
    model.set_output_embeddings(new_embeddings=output_embed_new)
    return model


def init_lora_model(
        model,
        r=32,
        alpha=64,
        dropout=0.1,
        target_modules=None,
        unfreeze_embedding=False
):
    """Initializes a peft model with lora layers"""
    # set q, k and v as default for LoRA layers
    if target_modules is None:
        target_modules = ["k", "v", "q"]

    # load lora config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )

    # inject lora layers and freeze all other parameters
    model = get_peft_model(model, peft_config)

    # unfreeze embedding layers if desired
    if unfreeze_embedding:
        for name, param in model.named_parameters():
            if 'shared' in name or 'lm_head' in name:
                # unfreeze base model's embedding layer
                param.requires_grad = True
                print(f'unfreeze {name}')

    model.print_trainable_parameters()
    return model


def tokenize_data(df, tokenizer, max_len):
    """Tokenizes a dataset and returns dictionary with input ids, attention mask and labels"""
    tokenized = df.map(lambda x: tokenizer(x, max_length=max_len, padding='max_length', truncation=True))
    preprocessed_df = [
      {
          'input_ids': torch.tensor(source.input_ids, dtype=torch.int64),
          'attention_mask': torch.tensor(source.attention_mask, dtype=torch.int8),
          'labels': torch.tensor(target.input_ids, dtype=torch.int64)
      }
      for source, target in tokenized[['input', 'output']].values.tolist()]
    return preprocessed_df


def prepare_data(file, tokenizer, max_len=128):
    data = pd.read_csv(file)
    data_preprocessed = tokenize_data(data, tokenizer, max_len)
    return data_preprocessed


def prepare_mt5_finetuning(
        model_id: str,
        finetuned_model_id: str,
        file_train: str,
        file_val=None,
        tokenizer_id=None,
        trim_model=False,
        learning_rate=1e-3,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        lora_target_modules=None,
        unfreeze_embedding=False,
        num_train_epochs=1,
        train_batch_size=16,
        push_to_hub=False,
        hf_token=None,
        hub_private_repo=True,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=50,
):
    """Loads a model, sets up lora modules and prepares Trainer-object for fine-tuning"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if hf_token is None and push_to_hub:
        push_to_hub = False
        print('No hub-token provided, set push_to_hub to False')

    if evaluation_strategy != "no" and file_val is None:
        evaluation_strategy = "no"
        print('No validation file provided, set evaluation_stratedy to "no"')

    model, tokenizer = load_model(model_id, tokenizer_id, hf_token)

    if trim_model:
        model = trim_mt5_model(model, tokenizer)
        
    model = init_lora_model(
        model,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules,
        unfreeze_embedding,
    )
    model.to(device)

    train_data = prepare_data(file_train, tokenizer)
    val_data = None if file_val is None else prepare_data(file_val, tokenizer)

    optimizer = Adafactor(model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False,
                          clip_threshold=1.0, decay_rate=0.0)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    # Create training arguments
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=train_batch_size,
        push_to_hub=push_to_hub,
        output_dir=finetuned_model_id,
        hub_token=hf_token,
        hub_private_repo=hub_private_repo,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_dir=f'logs',
        logging_strategy="steps",
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=save_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        lr_scheduler_type="constant",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        # compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    return trainer, tokenizer


def save_model(trainer, name, hf_token=None, hub_name=None, save_embedding_layer=False):
    """Saves a model locally and to the hub, if desired"""
    if hf_token is not None and hub_name is not None:
        trainer.model.push_to_hub(f'{hub_name}/{name}', token=hf_token, private=True, save_embedding_layers=save_embedding_layer)
    trainer.save_model(name)


def save_tokenizer(tokenizer, path, hf_token=None, hub_name=None):
    """Saves a tokenizer locally and to the hub, if desired"""
    if hf_token is not None and hub_name is not None:
        tokenizer.push_to_hub(f'{hub_name}/{path}', token=hf_token, private=True)
    tokenizer.save_pretrained(path)


def plot_loss(trainer):
    """Plot the train and eval loss of the training process"""
    history = pd.DataFrame(trainer.state.log_history)
    plt.plot(history['loss'])

    if history['eval_loss'].empty:
        eval_loss = history['eval_loss']
        xs = np.arange(len(eval_loss))
        series = np.array(eval_loss).astype(np.double)
        s1mask = np.isfinite(series)
        plt.plot(xs[s1mask], series[s1mask])

    plt.show()
    

def predict(ds, model, tokenizer):
    """Generates model predictions"""
    inputs = tokenizer(ds['input'], return_tensors='pt').to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {'sources': ds['input'], 'predictions': predictions[0], 'references': ds['output']}


def calculate_score(predictions_df, lang):
    """Calculates SARI, BLEU and BERTScore"""
    sari = corpus_sari(orig_sents=predictions_df['sources'],
                       sys_sents=predictions_df['predictions'],
                       refs_sents=[predictions_df['references']])
    bleu = corpus_bleu(sys_sents=predictions_df['predictions'], refs_sents=[predictions_df['references']])
    bert = corpus_bertscore(sys_sents=predictions_df['predictions'], refs_sents=[predictions_df['references']], lang=lang)
    return sari, bleu, bert


def evaluate_model(
        peft_model_id,
        base_model_id,
        file_test,
        save_dir,
        tokenizer_id=None,
        n=None,
        lang="de",
        hf_token=None,
        hub_name=None,
        resize_embedding=False,
):
    """Loads a peft model, generates predictions based on test file and calculates automatic evaluation scores."""
    if hf_token is not None and hub_name is not None:
        model, tokenizer = load_peft_model(
            peft_model_id=f'{hub_name}/{peft_model_id}', 
            base_model_id=base_model_id, 
            tokenizer_id=tokenizer_id, 
            hf_token=hf_token,
        )
    else:
        model, tokenizer = load_peft_model(
            peft_model_id=peft_model_id, 
            base_model_id=base_model_id, 
            tokenizer_id=tokenizer_id, 
            hf_token=hf_token,
        )
    model = torch.compile(model)

    test_data = load_dataset("csv", data_files=file_test)

    test_data["train"] = Dataset.from_dict(test_data["train"][:n])
    result = test_data.map(lambda x: predict(x, model, tokenizer), remove_columns=['input', 'output'])

    result_df = pd.DataFrame(result["train"])

    result_df.to_csv(f'{save_dir}/{peft_model_id}_prediction_results.csv', index=False, quoting=csv.QUOTE_ALL)
    sari, bleu, bert = calculate_score(result_df, lang)

    with open(f'{save_dir}/{peft_model_id}_scores.json', 'w') as f:
        json.dump({"sari": sari, "bleu": bleu, "bert_score": bert}, f)
        f.close()

    bert_score_p = bert[0]
    print(f'SARI: {sari:.2f} - BLEU: {bleu:.2f} - BERTScore Precision: {bert_score_p:.4f}')
    return sari, bleu, bert
