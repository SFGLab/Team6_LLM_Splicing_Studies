import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback, 
)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils import resample 
import numpy as np
import os
import logging
import time 
import gc 
import random
import json 

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil library not found. Memory usage logging will be limited. Install with 'pip install psutil'")

logger = logging.getLogger(__name__)


# --- Configuration Loading ---
def load_config(config_filepath="config.json"):
    """Loads configuration from a JSON file."""

    with open(config_filepath, 'r') as f:
        config_data = json.load(f) 
    
    log_level_str = config_data.get("logging_params", {}).get("log_level", "INFO").upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True) # force=True to reconfigure if already set
    logger.setLevel(numeric_level) 
    
    logger.info(f"Configuration loaded from {config_filepath}")
    return config_data

# --- Memory Logging Helper ---
def log_memory_usage(stage_name=""):
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory Usage [{stage_name}]: RSS={mem_info.rss / (1024**2):.2f} MB, VMS={mem_info.vms / (1024**2):.2f} MB")
    else:
        logger.debug(f"Memory Usage [{stage_name}]: psutil not available for detailed report.") # Changed to debug for less noise

# --- Custom Training Time Callback ---
class TrainingTimeCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_begin_time = None
        self.epoch_begin_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_begin_time = time.time()
        logger.info("Starting training...")
        log_memory_usage("Training Start")

    def on_train_end(self, args, state, control, **kwargs):
        if self.train_begin_time:
            total_training_time = time.time() - self.train_begin_time
            logger.info(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
        log_memory_usage("Training End")
        gc.collect()


    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_begin_time = time.time()
        logger.info(f"Starting Epoch {int(state.epoch + 1)}/{int(state.num_train_epochs)}")
        log_memory_usage(f"Epoch {int(state.epoch + 1)} Start")


    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_begin_time and self.train_begin_time:
            epoch_duration = time.time() - self.epoch_begin_time
            total_elapsed_time = time.time() - self.train_begin_time
            
            current_epoch_num = int(state.epoch + 1)
            total_epochs = int(state.num_train_epochs)
            
            avg_time_per_epoch = total_elapsed_time / current_epoch_num
            remaining_epochs = total_epochs - current_epoch_num
            estimated_remaining_time = remaining_epochs * avg_time_per_epoch
            
            logger.info(f"Epoch {current_epoch_num}/{total_epochs} finished in {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))}")
            logger.info(f"Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed_time))}")
            if remaining_epochs > 0:
                logger.info(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))}")
        log_memory_usage(f"Epoch {int(state.epoch + 1)} End")
        gc.collect()


# --- 1. Custom PyTorch Dataset ---
class JunctionSequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = sequences 
        self.labels = labels     
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):
        sequence = str(self.sequences[item_idx])
        label = int(self.labels[item_idx])

        encoding = self.tokenizer.encode_plus(
            sequence.upper(), add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,            
            return_attention_mask=True, return_tensors='pt',        
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 2. Load and Prepare Data Function ---
def read_sequences_from_file(filepath, label, expected_format, max_samples, random_state_val):
    log_memory_usage(f"Before processing {os.path.basename(filepath)}")
    sequences = []
    genes = []
    
    if not os.path.exists(filepath):
        logger.warning(f"Data file not found: {filepath}. Returning empty DataFrame.")
        return pd.DataFrame({'gene': [], 'sequence': [], 'label': []})
        
    relevant_line_count = 0
    with open(filepath, 'r') as f:
        for line_content in f:
            line_content = line_content.strip()
            if not line_content or line_content.startswith("#"):
                continue
            relevant_line_count += 1
    logger.info(f"File {filepath} contains {relevant_line_count} relevant lines.")
    log_memory_usage(f"After counting lines in {os.path.basename(filepath)}")

    lines_to_process = []
    if max_samples is not None and relevant_line_count > max_samples:
        logger.info(f"Sampling {max_samples} lines from {relevant_line_count} relevant lines in {filepath}.")
        random.seed(random_state_val) 
        
        indices_to_keep = sorted(random.sample(range(relevant_line_count), max_samples))
        
        current_relevant_line_idx = 0
        kept_line_idx_ptr = 0
        with open(filepath, 'r') as f:
            for line_content in f:
                line_content = line_content.strip()
                if not line_content or line_content.startswith("#"):
                    continue
                
                if kept_line_idx_ptr < len(indices_to_keep) and current_relevant_line_idx == indices_to_keep[kept_line_idx_ptr]:
                    lines_to_process.append(line_content)
                    kept_line_idx_ptr += 1
                current_relevant_line_idx += 1
                if kept_line_idx_ptr >= len(indices_to_keep): 
                    break
        logger.info(f"Selected {len(lines_to_process)} lines after sampling for {filepath}.")
    else:
        logger.info(f"Processing all {relevant_line_count} relevant lines from {filepath} (max_samples not exceeded or not set).")
        with open(filepath, 'r') as f:
            for line_content in f:
                line_content = line_content.strip()
                if not line_content or line_content.startswith("#"):
                    continue
                lines_to_process.append(line_content)
    log_memory_usage(f"After selecting/reading lines for {os.path.basename(filepath)}")

    for line_num, line in enumerate(lines_to_process):
        seq = ""
        gene_info = f"unknown_gene_line{line_num+1}" 
        if expected_format == 'gene_seq': 
            parts = line.split(';', 1)
            if len(parts) == 2:
                gene_info = parts[0]
                seq = parts[1]
            else:
                logger.warning(f"Line in {filepath} not in 'gene;sequence' format: '{line}'. Treating whole line as sequence.")
                seq = line 
        elif expected_format == 'seq_only': 
            seq = line
        else:
            raise ValueError(f"Unknown expected_format: {expected_format}")

        if seq: 
            sequences.append(seq.upper()) 
            genes.append(gene_info)
        else:
            logger.warning(f"Empty sequence found in {filepath} for line: '{line}'.")
    
    del lines_to_process 
    gc.collect()

    log_memory_usage(f"Before creating DataFrame for {os.path.basename(filepath)}")
    df = pd.DataFrame({'gene': genes, 'sequence': sequences, 'label': label})
    log_memory_usage(f"After creating DataFrame for {os.path.basename(filepath)} (Size: {len(df)})")
    if PSUTIL_AVAILABLE and not df.empty:
         logger.info(f"DataFrame memory usage for {os.path.basename(filepath)}: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    return df


def load_and_prepare_datasets(config, tokenizer):
    """
    Loads data based on config, combines, handles imbalance, splits, and creates Datasets.
    """
    neojunction_filepath = config['data_paths']['neojunction_file_path']
    normal_junction_filepath = config['data_paths']['normal_junction_file_path']
    max_samples_per_class = config['data_processing'].get('max_samples_per_class') 
    max_seq_len = config['data_processing']['max_seq_length']
    resample_strategy = config['data_processing'].get('resample_strategy_train_set')
    test_size = config['data_processing']['validation_set_test_size']
    random_state_val = config['training_params']['random_seed']
    input_file_format = config['data_processing']['input_file_format']

    log_memory_usage("Start of load_and_prepare_datasets")
    
    logger.info(f"Loading neojunction data from: {neojunction_filepath}, limit: {max_samples_per_class}")
    df_neo = read_sequences_from_file(neojunction_filepath, label=1, 
                                      expected_format=input_file_format, 
                                      max_samples=max_samples_per_class,
                                      random_state_val=random_state_val) 
    log_memory_usage("After loading neojunctions")
    
    logger.info(f"Loading normal junction data from: {normal_junction_filepath}, limit: {max_samples_per_class}")
    df_normal = read_sequences_from_file(normal_junction_filepath, label=0, 
                                         expected_format=input_file_format, 
                                         max_samples=max_samples_per_class,
                                         random_state_val=random_state_val)
    log_memory_usage("After loading normal junctions")

    if df_neo.empty and df_normal.empty:
        logger.error("Both neojunction and normal junction files are empty or not found (or resulted in empty data after sampling).")
        logger.info(f"Creating dummy data for demonstration as input files are missing/empty.")
        num_dummy_samples = min(max_samples_per_class if max_samples_per_class else 20, 20) 
        dummy_sequences = []
        for _ in range(num_dummy_samples): 
            chars = ['A', 'C', 'G', 'T']
            seq_len_val = np.random.randint(max(10, max_seq_len - 50), max_seq_len + 50) 
            seq = "".join(np.random.choice(chars, size=seq_len_val))
            dummy_sequences.append(seq)
        
        df_combined = pd.DataFrame({
            'gene': [f'dummy_gene_{i}' for i in range(num_dummy_samples * 2)],
            'sequence': dummy_sequences * 2, 
            'label': [1]*num_dummy_samples + [0]*num_dummy_samples 
        })
        logger.info(f"Created dummy combined DataFrame with {len(df_combined)} entries.")
        if not os.path.exists(neojunction_filepath):
             with open(neojunction_filepath, 'w') as f_dummy_neo:
                 for _, row in df_combined[df_combined['label']==1].iterrows():
                     f_dummy_neo.write(f"{row['gene']};{row['sequence']}\n")
             logger.info(f"Created dummy {neojunction_filepath}")
        if not os.path.exists(normal_junction_filepath):
            with open(normal_junction_filepath, 'w') as f_dummy_normal:
                 for _, row in df_combined[df_combined['label']==0].iterrows():
                     f_dummy_normal.write(f"{row['gene']};{row['sequence']}\n")
            logger.info(f"Created dummy {normal_junction_filepath}")
    elif df_neo.empty:
        df_combined = df_normal
    elif df_normal.empty:
        df_combined = df_neo
    else:
        df_combined = pd.concat([df_neo, df_normal], ignore_index=True)
    
    del df_neo, df_normal 
    gc.collect()
    log_memory_usage("After combining DataFrames")

    logger.info(f"Total combined records (after per-class sampling if applied): {len(df_combined)}")
    df_combined = df_combined.dropna(subset=['sequence', 'label'])
    df_combined['sequence'] = df_combined['sequence'].astype(str)
    df_combined['label'] = df_combined['label'].astype(int)
    logger.info(f"Combined records after initial cleaning: {len(df_combined)}")

    if len(df_combined) == 0:
        raise ValueError("No valid data found after combining and cleaning. Please check your input files and sampling limits.")
    if not (df_combined['label'].isin([0, 1])).all():
        raise ValueError("Labels must be 0 or 1 after combining files.")

    logger.info(f"Initial label distribution of combined data:\n{df_combined['label'].value_counts(normalize=True)}")
    log_memory_usage("Before train/test split")

    min_samples_for_split = 10 
    if len(df_combined) < min_samples_for_split or len(df_combined['label'].unique()) < 2:
        train_df = df_combined.copy()
        val_df = pd.DataFrame(columns=train_df.columns) 
    else:
        train_df, val_df = train_test_split(
            df_combined, test_size=test_size,
            stratify=df_combined['label'], random_state=random_state_val
        )
    del df_combined 
    gc.collect()
    log_memory_usage("After train/test split")

    logger.info(f"Training set size before resampling: {len(train_df)}")
    if not train_df.empty:
         logger.info(f"Training set label distribution before resampling:\n{train_df['label'].value_counts(normalize=True)}")

    if resample_strategy and not train_df.empty and len(train_df['label'].unique()) > 1:
        logger.info(f"Applying resampling strategy: {resample_strategy} to the training set.")
        label_counts = train_df['label'].value_counts()
        
        if len(label_counts) < 2 or label_counts.min() == 0:
            logger.warning("Not enough classes or samples in one class in training set to resample. Skipping resampling.")
        else:
            minority_class_label = label_counts.idxmin()
            majority_class_label = label_counts.idxmax()
            
            df_majority = train_df[train_df['label'] == majority_class_label]
            df_minority = train_df[train_df['label'] == minority_class_label]

            if resample_strategy == 'oversample':
                if not df_minority.empty:
                    df_minority_resampled = resample(df_minority,
                                                     replace=True, 
                                                     n_samples=len(df_majority), 
                                                     random_state=random_state_val)
                    train_df_resampled = pd.concat([df_majority, df_minority_resampled])
                else: 
                    train_df_resampled = train_df 
            elif resample_strategy == 'undersample':
                if not df_majority.empty:
                    df_majority_resampled = resample(df_majority,
                                                     replace=False, 
                                                     n_samples=len(df_minority), 
                                                     random_state=random_state_val)
                    train_df_resampled = pd.concat([df_minority, df_majority_resampled])
                else:
                    train_df_resampled = train_df
            else:
                logger.warning(f"Unknown resample_strategy: {resample_strategy}. No resampling applied.")
                train_df_resampled = train_df
            
            train_df = train_df_resampled.sample(frac=1, random_state=random_state_val).reset_index(drop=True) 
            logger.info(f"Training set size after resampling: {len(train_df)}")
            logger.info(f"Training set label distribution after resampling:\n{train_df['label'].value_counts(normalize=True)}")
            del df_majority, df_minority, train_df_resampled 
            gc.collect()
        log_memory_usage("After resampling training set")


    logger.info(f"Final training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    if train_df.empty:
        raise ValueError("Training dataset is empty after processing.")

    log_memory_usage("Before creating PyTorch Datasets")
    train_dataset = JunctionSequenceDataset(
        sequences=train_df.sequence.to_numpy(), labels=train_df.label.to_numpy(),
        tokenizer=tokenizer, max_len=max_seq_len
    )
    val_dataset = None
    if not val_df.empty:
        val_dataset = JunctionSequenceDataset(
            sequences=val_df.sequence.to_numpy(), labels=val_df.label.to_numpy(),
            tokenizer=tokenizer, max_len=max_seq_len
        )
    log_memory_usage("After creating PyTorch Datasets")
    del train_df, val_df 
    gc.collect()
    return train_dataset, val_dataset

# --- 3. Metrics Computation Function ---
def compute_classification_metrics(p: EvalPrediction):
    preds_logits = p.predictions
    preds_probs = torch.softmax(torch.tensor(preds_logits), dim=-1).numpy()
    hard_preds = np.argmax(preds_logits, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, hard_preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, hard_preds)
    auc_roc = -1.0 
    if len(np.unique(labels)) > 1 and preds_probs.shape[1] == 2: 
        try: auc_roc = roc_auc_score(labels, preds_probs[:, 1]) 
        except ValueError as e: logger.warning(f"Could not compute AUC-ROC: {e}.")
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auc_roc': auc_roc}

# --- 4. Main Fine-Tuning Function ---
def run_finetuning(config):
    """Main function to orchestrate the fine-tuning process, using config."""
    logger.info("Starting SpliceBERT fine-tuning for neojunction classification...")
    log_memory_usage("Run Finetuning Start")

    pretrained_model_path = config['model_paths']['pretrained_model_path']
    tokenizer_path = config['model_paths']['tokenizer_path']
    output_dir = config['training_params']['output_dir']
    # max_seq_length is used within load_and_prepare_datasets, fetched from config there
    num_train_epochs = config['training_params']['num_train_epochs']
    per_device_train_batch_size = config['training_params']['per_device_train_batch_size']
    per_device_eval_batch_size = config['training_params']['per_device_eval_batch_size']
    learning_rate = config['training_params']['learning_rate']
    random_seed = config['training_params']['random_seed']


    if torch.cuda.is_available():
        logger.info(f"CUDA available! Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available. Training on CPU (this will be much slower).")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=False, local_files_only=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True); return
    log_memory_usage("Tokenizer Loaded")

    try:
        train_dataset, val_dataset = load_and_prepare_datasets(config, tokenizer) 
    except Exception as e:
        logger.error(f"Failed to load/prepare data: {e}", exc_info=True); return
    
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Training dataset empty. Aborting."); return
    
    do_eval_runtime = val_dataset is not None and len(val_dataset) > 0 
    if not do_eval_runtime: logger.warning("Validation dataset empty. Evaluation during training and loading best model at end will be skipped.")
    log_memory_usage("Datasets Prepared")

    logger.info(f"Loading pre-trained model from: {pretrained_model_path}")
    try:
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_path, num_labels=2, local_files_only=True)
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True); return
    log_memory_usage("Model Loaded")
    
    steps_per_epoch = len(train_dataset) // per_device_train_batch_size if per_device_train_batch_size > 0 and len(train_dataset) > 0 else 1
    logging_steps_val = max(1, int(0.1 * steps_per_epoch)) if steps_per_epoch > 0 else 10 
    warmup_steps_val = max(1, int(0.1 * steps_per_epoch * num_train_epochs)) if steps_per_epoch > 0 else 50 

    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps_val,
        "weight_decay": 0.01,
        "logging_dir": f'{output_dir}/logs',
        "logging_steps": logging_steps_val,
        "save_total_limit": 2,
        "report_to": "tensorboard",
        "fp16": torch.cuda.is_available(),
        "seed": random_seed,
    }

    if do_eval_runtime:
        training_args_dict["evaluation_strategy"] = "epoch"
        training_args_dict["save_strategy"] = "epoch" 
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "f1"
        training_args_dict["greater_is_better"] = True
    else: 
        training_args_dict["evaluation_strategy"] = "no" 
        training_args_dict["save_strategy"] = "epoch" 
        training_args_dict["load_best_model_at_end"] = False


    training_args = TrainingArguments(**training_args_dict)
    
    time_callback = TrainingTimeCallback()
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset if do_eval_runtime else None,
        compute_metrics=compute_classification_metrics if do_eval_runtime else None,
        callbacks=[time_callback] 
    )
    logger.info("Starting fine-tuning training loop...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True); return

    if do_eval_runtime:
        logger.info("Evaluating final model on validation set...")
        eval_results = trainer.evaluate()
        logger.info(f"Final validation results: {eval_results}")

    final_model_path = os.path.join(output_dir, "best_model")
    logger.info(f"Saving model to {final_model_path}...")
    trainer.save_model(final_model_path) 
    tokenizer.save_pretrained(final_model_path)
    logger.info("Fine-tuning process complete.")

if __name__ == '__main__':
    config_file = "config.json"
    config = load_config(config_file) 
    
    random_seed_val = config['training_params']['random_seed']
    torch.manual_seed(random_seed_val)
    np.random.seed(random_seed_val)
    random.seed(random_seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed_val)


    logger.info("--- SpliceBERT Fine-Tuning for Neojunction Classification (Config-Driven) ---")
    
    pretrained_model_path_conf = config['model_paths']['pretrained_model_path']
    tokenizer_path_conf = config['model_paths']['tokenizer_path']
    neojunction_file_path_conf = config['data_paths']['neojunction_file_path']
    normal_junction_file_path_conf = config['data_paths']['normal_junction_file_path']


    if not os.path.exists(pretrained_model_path_conf) or not os.path.isdir(pretrained_model_path_conf):
        logger.warning(f"PRETRAINED_MODEL_PATH '{pretrained_model_path_conf}' does not exist. Creating dummy structure.")
        os.makedirs(pretrained_model_path_conf, exist_ok=True)
        with open(os.path.join(pretrained_model_path_conf, "config.json"), "w") as f:
            f.write('{"model_type": "bert", "num_hidden_layers": 1, "hidden_size": 10, "intermediate_size": 10, "num_attention_heads": 1, "vocab_size": 30000, "max_position_embeddings": 512}')
        logger.warning("Dummy config.json created. You NEED actual SpliceBERT model files for training.")

    if not os.path.exists(tokenizer_path_conf) or not os.path.isdir(tokenizer_path_conf):
        logger.warning(f"TOKENIZER_PATH '{tokenizer_path_conf}' does not exist. Creating dummy structure.")
        os.makedirs(tokenizer_path_conf, exist_ok=True)
        with open(os.path.join(tokenizer_path_conf, "tokenizer_config.json"), "w") as f:
            f.write('{"tokenizer_class": "AutoTokenizer", "do_lower_case": false, "model_max_length": 512}') 
        with open(os.path.join(tokenizer_path_conf, "vocab.txt"), "w") as f: 
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nA\nC\nG\nT\nN\n")
        logger.warning("Dummy tokenizer files created. You NEED actual SpliceBERT tokenizer files and config.")
    
    # The load_and_prepare_datasets function will create dummy data files if they are missing
    # and paths are valid. We just log a warning here.
    for data_file_path in [neojunction_file_path_conf, normal_junction_file_path_conf]:
        if not os.path.exists(data_file_path):
            logger.warning(f"Data file '{data_file_path}' not found. The script will attempt to create a dummy version if needed during data loading.")

    logger.info(f"Running with configuration from: {config_file}")
    logger.info(json.dumps(config, indent=4)) # Log the loaded config
    logger.info("-" * 60)
    
    run_finetuning(config)