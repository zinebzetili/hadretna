train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Convert the sample rate of every audio file using the cast_column function
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load the feature extractor, tokenizer, and processor from the pretrained model
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Arabic", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Arabic", task="transcribe")

# Function to prepare the dataset by extracting features and encoding labels
def prepare_dataset(examples):
    # Compute log-Mel input features from input audio array
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    del examples["audio"]
    
    # Encode target text to label IDs
    sentences = examples["sentence"]
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["sentence"]
    return examples

# Apply the prepare_dataset function to the training and testing datasets
train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

# Define a data collator for padding inputs and labels appropriately
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences and pad them to max length
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS token is appended in previous tokenization step, cut it here
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Load the metric for evaluation
metric = evaluate.load("wer")

# Function to compute metrics (WER) during evaluation
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad token ID
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Load the pre-trained Whisper model for conditional generation
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-en",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=50,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Initialize the trainer with the model, arguments, datasets, data collator, and metrics
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Start the model training
trainer.train()
