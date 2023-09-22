from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import json
import os
import gc
import re

### Functions
def read_jsonl_lazy(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def json_conversations_to_list(file_path): # Dataset version 1: populates the "conversations" variable with ["text"(conversations), "reversed", "source", "score"]
    if dataset_version == 1:
        for item in read_jsonl_lazy(file_path):
            text = "\n|`|\n".join(item.get("conversations", ["UH OH, NO CONVO"]))
            tags = item.get("tags", [])
            score = item.get("dataset_quality")
            if not score:
                score = item.get("score", 1)
            if "reversed" in tags:
                reversed = True
            else:
                reversed = False
            yield {
                "text": text,
                "reversed": reversed,
                "source": item.get("source", dataset_name),
                "score": int(score)
            }
    else:
        for item in read_jsonl_lazy(file_path):# Dataset version 2: populates the "conversations" variable with ["sys"(system prompt/context), "text"(conversations), "source", "score", "synthetic", "factual", "tags"]
            text = "\n|`|\n".join(item.get("conversations", ["UH OH, NO CONVO"]))
            tags = item.get("tags", [])
            quality = item.get("score")
            if not quality:
                quality = item.get("dataset_quality", 1)
            if item.get("reversed", False):
                tags.append("reversed")
            yield {
                "sys": item.get("init", ""),
                "text": text,
                "source": item.get("source", dataset_name),
                "score": int(quality),
                "synthetic": item.get("synthetic_origin", False),
                "factual": item.get("likely_factual", False),
                "tags": tags # All tags are just passed through to the final dataset, and do not make a difference in the uniqeness calculations
            }

def dataset_finalizer(conversations):
    if dataset_version == 1:
        conversations_to_save = [{
                "conversations": conv["text"].split("\n|`|\n"),
                "reversed": conv["reversed"],
                "source": conv["source"],
                "score": conv["score"]
            } for conv in conversations ]
    else:
        conversations_to_save = [{
                "init": conv["sys"],
                "conversations": conv["text"].split("\n|`|\n"),
                "source": conv["source"],
                "dataset_quality": conv["score"],
                "synthetic_origin": conv["synthetic"],
                "likely_factual": conv["factual"],
                "tags": conv["tags"]
            } for conv in conversations ]
    return conversations_to_save # Output: The final dataset in the desired format

def dataset_dumper(conversations):
    if script_mode == 1:
        file = dataset_path + "_Deduped_and_Filtered.jsonl"
    elif script_mode == 2:
        file = dataset_path + "_Deduped.jsonl"
    elif script_mode == 3:
        file = dataset_path + "_Filtered.jsonl"
    with open(file, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    print(">Saved!<")

def json_conversations_to_list_to_embed(file_path):
    for item in read_jsonl_lazy(file_path):
        text = "\n".join(item["conversations"])
        yield text # Output: All conversations as singular strings

def calculate_metrics(conversations):
    total_tokens = 0
    total_words = 0
    total_convos = 0
    
    for conversation in tqdm(conversations, desc=f">Calculating conversation", leave=False):
        text = " ".join(conversation["conversations"])
        tokens = tokenizer(text)['input_ids']
        words = text.split()
        
        total_tokens += len(tokens)
        total_words += len(words)
        total_convos += 1
    
    return total_tokens, total_words, total_convos

def approximate_final_convo_amount(conversations_count):
    remaining_factor = 1 - (X / 100) - (Y / 100) + (X * Y / 10000)
    estimated_remaining_conversations = conversations_count * (remaining_factor ** n_iterations)
    print(f">Estimate of remaining conversations after {n_iterations} iterations: ~{int(estimated_remaining_conversations)}<")

def tokenize_batch(batch, start_idx, tokenizer, pattern):  # Input: List of conversations, starting index in the original dataset, maximum token length, tokenizer, and text-cleaning pattern
    embeddings_to_make_local = []
    for idx, convo in enumerate(batch):
        # Text Preprocessing: Removing unwanted characters using regex pattern and splitting conversation into chunks of words to stay under the tokenizer's max sequence length
        cleaned_convo = pattern.sub('', convo)
        cleaned_convo = ' '.join(cleaned_convo.split())
        words = cleaned_convo.split(" ")
        word_chunks = [' '.join(words[i:i + (max_seq_len // 3)]) for i in range(0, len(words), (max_seq_len // 3))] # Split words into chunks of max_seq_len//3 words each
        word_chunks = [word_chunks[0]] + [f" {chunk}" for chunk in word_chunks[1:]]
        
        # Tokenization: Tokenizing the word chunks into tokens
        tokenized_word_chunks = [tokenizer.encode(chunk, add_special_tokens=False, truncation=True, max_length=max_seq_len) for chunk in word_chunks]
        
        # Aggregating Tokens: Concatenating the tokens from all chunks into a single list for the given conversation
        all_tokens = [token for chunk in tokenized_word_chunks for token in chunk]
        embeddings_to_make_local.append({'tokens': all_tokens, 'original_idx': start_idx+idx})
    
    return embeddings_to_make_local  # Output: List of dictionaries with tokenized text and original index

def generate_embeddings(conversations, iteration):  # Input: List of conversations, with all turns squashed into one line of text. Iteration: chunk of the dataset that is being processed. max_length - tokenizer's sequence length
    all_embeddings = []
    idx_to_embeddings = {}
    embeddings_to_make = []
    
    # Threaded Tokenization: Tokenizing the text across multiple threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, len(conversations), pre_process_batch_size):
            batch = conversations[i:i + pre_process_batch_size]
            futures.append(executor.submit(tokenize_batch, batch, i, tokenizer, pattern))
        for future in tqdm(futures, desc=f">Pre-processing text", leave=False, postfix = f"Chunk {iteration + 1}/{chunks_to_embed_count}"):
            embeddings_to_make.extend(future.result())

    # Embedding Generation: Looping through tokenized batches to generate embeddings
    for i in tqdm(range(0, len(embeddings_to_make), embeddings_batch_size), desc=f">Generating embeddings", postfix = f"Chunk {iteration + 1}/{chunks_to_embed_count}"):
        batch_to_process = embeddings_to_make[i:i + embeddings_batch_size]
        tensor_list = []
        attention_mask_list = []

        # Batch Preparation: Prepare input tensors and attention masks for model inference
        for item in batch_to_process:
            tokens = item['tokens']
            original_idx = item['original_idx']
            tensor_list.append(tokens[:max_seq_len])
            attention_mask_list.append([1] * min(len(tokens), max_seq_len))

            # Handling Overlength Tokens: If the amount of tokens exceed max_length, store them in idx_to_embeddings
            if len(tokens) > max_seq_len:
                if original_idx not in idx_to_embeddings:
                    idx_to_embeddings[original_idx] = []
        
        # Tensor Padding: Padding the tensors and attention masks to be of equal length
        max_len = max(min(len(item), max_seq_len) for item in tensor_list)
        tensor_list = np.array([np.pad(a, (0, max_len - len(a)), 'constant') for a in tensor_list])
        attention_mask_list = np.array([np.pad(a, (0, max_len - len(a)), 'constant') for a in attention_mask_list])
        
        # Convert to PyTorch tensor and move to device
        tensor_batch = torch.tensor(tensor_list, dtype=torch.long).to(device)
        attention_mask_batch = torch.tensor(attention_mask_list, dtype=torch.long).to(device)

        # Model Inference: Generating embeddings from the model
        with torch.no_grad():
            output = model(input_ids=tensor_batch, attention_mask=attention_mask_batch).last_hidden_state.mean(dim=1).cpu().numpy()

        # Saving Embeddings: Store the generated embeddings
        for j, item in enumerate(batch_to_process):
            original_idx = item['original_idx']
            if len(item['tokens']) <= max_seq_len:
                all_embeddings.append((original_idx, output[j]))
            else:
                idx_to_embeddings[original_idx].append(output[j])

    # Finalize Overlength Embeddings: Average the embeddings of the tokens that exceeded max_length
    for idx, chunks in idx_to_embeddings.items():
        averaged_embedding = np.mean(np.vstack(chunks), axis=0)
        all_embeddings.append((idx, averaged_embedding))
    
    # Sorting Embeddings: Sort all generated embeddings by their original index
    all_embeddings.sort(key=lambda x: x[0])
    final_embeddings = np.array([emb[1] for emb in all_embeddings])
    gc.collect()
    
    return final_embeddings  # Output: NumPy array of any-dimensional embeddings, in the order of their original conversations

def compute_batch_similarity(start, end, data_quality_all, normalized_embeddings, leave_one_out_aggregate_all, weights):
    batch_similarities = np.zeros(end - start)
    leave_one_out_aggregate_all = leave_one_out_aggregate_all[start:end]
    data_quality_all = data_quality_all[start:end]

    for i, embedding in enumerate(normalized_embeddings[start:end]):
        leave_one_out_aggregate = leave_one_out_aggregate_all[i]
        data_quality = data_quality_all[i]

        weighted_similarity = np.dot(embedding * weights, leave_one_out_aggregate * weights)
        basic_similarity = np.dot(embedding, leave_one_out_aggregate)

        cosine_similarity = alpha * basic_similarity + (1 - alpha) * weighted_similarity

        euclidean_dist = np.linalg.norm(embedding - leave_one_out_aggregate)

        combined_similarity = (beta * cosine_similarity + (1 - beta) * (1 / (1 + euclidean_dist)))/(1 + ((data_quality - 1) * quality_factor)/100)
        batch_similarities[i] = combined_similarity
    return batch_similarities # Output: Array of floats from 0 to 1. 0 being most unique, 1 - least.

def compute_uniqueness(embeddings): # Input: NumPy array of any-dimensional embeddings, in the order of their original conversations
    total_embeddings = len(embeddings)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    aggregate_embedding = np.mean(normalized_embeddings, axis=0)
    aggregate_embedding /= np.linalg.norm(aggregate_embedding)
    precomputed_aggregate = total_embeddings * aggregate_embedding

    weights = np.abs(aggregate_embedding)
    similarities = np.zeros(total_embeddings)
    leave_one_out_aggregate_all = []
    data_quality_all = []

    for line in conversations:
        data_quality_all.append(line.get("score", 1))

    for embedding in normalized_embeddings:
        leave_one_out_aggregate = (precomputed_aggregate - embedding) / (total_embeddings - 1)
        leave_one_out_aggregate /= np.linalg.norm(leave_one_out_aggregate)
        leave_one_out_aggregate_all.append(leave_one_out_aggregate)

    ranges = [(i, min(i + uniqueness_batch_size, total_embeddings)) for i in range(0, total_embeddings, uniqueness_batch_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_batch_similarity, start, end, data_quality_all, normalized_embeddings, leave_one_out_aggregate_all, weights) for start, end in ranges]

        for i, future in enumerate(futures):
            start, end = ranges[i]
            similarities[start:end] = future.result()

    uniqueness_values = 1 - similarities
    return uniqueness_values # Output: Array of floats from 0 to 1. 1 being most unique, 0 - least. Corresponding in position to their respective embedding/conversation

def filter_conversations(uniqueness_values, X, Y): # Input: Array of floats from 0 to 1. 1 being most unique, 0 - least. Corresponding in position to their respective embedding/conversation
    sorted_indices = np.argsort(uniqueness_values)
    if len(sorted_indices) <= 4:
        return sorted_indices

    # Remove bottom X% least unique conversations
    bottom_X_percent = int(len(sorted_indices) * X / 100)
    if bottom_X_percent == 0 and X != 0:
        bottom_X_percent = 1
    sorted_indices = sorted_indices[bottom_X_percent:] if bottom_X_percent > 0 else sorted_indices

    # Remove top Y% unique conversations
    top_Y_percent = int(len(sorted_indices) * Y / 100)
    if top_Y_percent == 0 and Y != 0:
        top_Y_percent = 1
    if top_Y_percent > 0:
        sorted_indices = list(reversed(sorted_indices))
        selected_indices = sorted_indices[top_Y_percent:]
        selected_indices = list(reversed(selected_indices))
    else: selected_indices = sorted_indices
    if preserve_original_order:
        selected_indices.sort()
    return selected_indices # Output: List of indexes as integers



### Main Logic
## Parameters
# Path, model, device, and script mode
file_path = "Path:/To/Your/Dataset.jsonl" # Example line of a supported dataset: `{"init": "Some system message", "conversations": ["AAAA", "aaaaaaa", "AAAAA"], "source": "aaa_dataset", "dataset_quality": 2, "synthetic_origin": false, "likely_factual": false, "tags": ["aaa"]}`."Conversations" follows a turn-based format, with no turn discriminators like "User:"/"Assistant:"/etc..
embed_model = "thenlper/gte-large" # gte-large is a pretty big model, which makes 1024 dim. embeddings, consider smaller embed models if you value time>quality
device = "cuda" # "cuda", "cpu", "mps"
script_mode = 1 # 1 - Dedupe and iteratievely filter out convos. 2 - Only dedupe. 3 - Only filter. 4 - Only embeddings

# Performance parameters
chunk_size = 100000 # Number of convos per chunk, if the dataset is big (significantly lessens the amount of Ram used when making embeddings)
num_threads = 8 # Threads to use for any multithreaded workloads
pre_process_batch_size = 8 # Batch size for pre-processing text for embeddings
embeddings_batch_size = 100 # Batch size when making embeddings (will use a lot more Vram/Ram)
uniqueness_batch_size = 1024 # Batch size when calculating uniqueness of conversations

# Configuration
dataset_version = 2 # 1 - Old style datasets. 2 - New style datasets (with sys prompt, and tags)
save_deduped = True # Save the deduped dataset and its embeddings?
use_deduped = False # Use the already deduped dataset and embeddings instead of the original to save time?
print_final_stats = True # Output the metrics of the final dataset?
early_approximation = True # Approximate the final number of convos, even before deduping?

# Deduping configuration
P = 10 # Save 1/P most unique conversations in a group of near-duplicates (set P to 0, to save only 1 most unique conversation out of every group)
match_threshold = 10 # Conversations that share match_threshold or more beginning words will be considered near-duplicates, and will be subject to deduplication

# Filtering configuration
n_iterations = 10 # Number of filtering iterations
X = 2.5 # Will delete the bottom X% least unique conversations each iteration
Y = 0 # Will delete the top Y% most unique conversations each iteration
alpha = 0.5 # Weight between basic and weighted similarities, 0.9 = (0.9 basic + 0.1 weighted) = cosine similarity
beta = 0.4 # Weight between cosine similarity and euclideian distance, 0.8 = 0.8 cosine + 0.2 euclidean
quality_factor = 1 # How much "dataset_quality" or "score" biases higher ranked convos to be considered more unique (0 = no impact, 2 = double the impact)
preserve_original_order = True # Save conversations in the order that they originally were? False - sort by uniqueness, bottom - most unique, top - least



## Logic execution
# Bad settings handling
if script_mode not in [1, 2, 3, 4]:
    print(f">Invalid script mode!<\n>1 - Dedupe and iteratievely filter out convos. 2 - Only dedupe. 3 - Only filter. 4 - Only embeddings<\n>Error was here: script_mode = {script_mode}<")
    exit()
if dataset_version not in [1, 2]:
    print(f">Invalid dataset version!<\n>1 - Old style datasets. 2 - New style datasets(with sys prompt, and tags)<\n>Error was here: dataset_version = {dataset_version}<")
    exit()
if type(n_iterations) is not int or n_iterations <= 0:
    print(f">Invalid number of iterations!<\n>Error was here: n_iterations = {n_iterations}<")
    exit()
if not os.path.exists(file_path) or not os.path.splitext(file_path)[1] == ".jsonl":
    print(f">Specified dataset doesn't exist or isn't the right file type! It must be a .jsonl file!<")
    exit()
if device == "cuda" and not torch.cuda.is_available():
    print(">Device was selected as 'Cuda', but no Cuda devices are available!<\n>Switching 'device' to 'cpu'<")
    device = "cpu"

# Misc variables
max_seq_len = 512 # Sequence length of the model and tokenizer, change it accordingly, if you're using some other embedding model
use_deduped_successful = False
final_convo_amount_showed = False
conversations = []
dataset_name = os.path.splitext(os.path.basename(file_path))[0]
dataset_directory = os.path.dirname(file_path)
dataset_path = f"{dataset_directory}\{dataset_name}"
embeddings_path = f"{dataset_path}_embeddings.npy"
deduped_file = f"{dataset_path}_Deduped.jsonl"
deduped_embeds_path = f"{dataset_path}_Deduped_embeddings.npy"

# Initialize the hardware and model
tokenizer = AutoTokenizer.from_pretrained(embed_model)
model = AutoModel.from_pretrained(embed_model)
model.to(device)

# Makin embeds
if os.path.exists(deduped_embeds_path) and os.path.exists(deduped_file) and use_deduped:
    print(">Deduped embeddings file found. Loading...<")
    embeddings = np.load(deduped_embeds_path)
    use_deduped_successful = True
elif os.path.exists(embeddings_path):
    print(">Embeddings file found. Loading...<")
    embeddings = np.load(embeddings_path)
else:
    print(f">Embeddings file of the dataset not found<\n>Making embeddings for {dataset_name}.jsonl...<")
    conversations_to_embed = list(json_conversations_to_list_to_embed(file_path))
    chunks_to_embed = [conversations_to_embed[i:i + chunk_size] for i in range(0, len(conversations_to_embed), chunk_size)]
    chunks_to_embed_count = len(chunks_to_embed)

    if early_approximation:
        approximate_final_convo_amount(len(conversations_to_embed))
        final_convo_amount_showed = True

    embeddings = []
    pattern = re.compile(r'\b(?:is|to|that|the|and|in|by|at|for|a|an)\b', re.IGNORECASE)# Remove function words, as they contain almost no lexical meaning, and thus provide almost no semantic value to the conversation
    
    for i, chunk in enumerate(chunks_to_embed):
        embeddings.extend(generate_embeddings(chunk, i))
        chunks_to_embed[i] = ""
        gc.collect()

    print(f">Total embeddings generated: {len(embeddings)}<\n>Saving embeddings to {dataset_name}_embeddings.npy...<")
    np.save(embeddings_path, np.array(embeddings))

# Commencing the dedupment
if script_mode in [1, 2]:
    if os.path.exists(deduped_file) and use_deduped and not script_mode == 2:
        print(">Found deduped dataset, skipping deduping...<")
        conversations = list(json_conversations_to_list(deduped_file))
    else:
        if not conversations:
            conversations = list(json_conversations_to_list(file_path))
        print(">Deduping in process...<")
        conversation_groups = defaultdict(list)

        if early_approximation and not final_convo_amount_showed:
            approximate_final_convo_amount(len(conversations))
            final_convo_amount_showed = True

        for idx, conv in enumerate(conversations):
            start_words = ' '.join(conv['text'].replace("\n|`|\n", " ").split()[:match_threshold])
            conversation_groups[start_words].append({'index': idx})

        conversations_to_save = []
        for start_words, group in tqdm(conversation_groups.items(), desc=f">Processing conversations", leave=False):
            if len(group) == 1:
                conversations_to_save.append(group[0]['index'])
                continue

            group_embeddings = [embeddings[item['index']] for item in group]
            group_uniqueness_values = compute_uniqueness(np.array(group_embeddings))
            if P == 0:
                n_indices_to_select = 1
            else:
                n_indices_to_select = len(group_uniqueness_values) // P
            sorted_unique_indices = np.argsort(group_uniqueness_values)
            most_unique_index = sorted_unique_indices[:n_indices_to_select]
            
            for i, item in enumerate(group):
                if i in most_unique_index:
                    conversations_to_save.append(item['index'])

        del conversation_groups
        gc.collect()
        conversations = [conversations[i] for i in conversations_to_save]
        embeddings = [embeddings[i] for i in conversations_to_save]

        if save_deduped and script_mode != 2:
            print(">Saving deduped dataset and embeddings...<")
            np.save(deduped_embeds_path, np.array(embeddings))
            deduped_conversations = dataset_finalizer(conversations)
            
            with open(f"{dataset_path}_Deduped.jsonl", 'w', encoding='utf-8') as f:
                for conversation in deduped_conversations:
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
            del deduped_conversations
            print(">Saved!<")

        print(">Deduping done<")
        gc.collect()

# Filtering conversations
if script_mode in [1, 3]:
    print(">Filtering in process...<")
    if os.path.exists(deduped_file) and use_deduped and not conversations:
        conversations = list(json_conversations_to_list(deduped_file))
    elif not conversations:
        conversations = list(json_conversations_to_list(file_path))

    conversations_count = len(conversations)
    approximate_final_convo_amount(conversations_count)
    uniqueness_values = compute_uniqueness(np.array(embeddings))

    for i in tqdm(range(n_iterations), desc=">Filtering Iterations"):
        indexes_to_save = filter_conversations(uniqueness_values, X, Y)
        conversations = [conversations[i] for i in indexes_to_save]
        embeddings = [embeddings[i] for i in indexes_to_save]
        uniqueness_values = compute_uniqueness(np.array(embeddings))

    gc.collect()
    print(">Filtering done<")

# Save the filtered conversations
if script_mode in [1, 2, 3]:
    final_conversations = dataset_finalizer(conversations) # Converts conversations into the desired final format

if script_mode == 1 or use_deduped_successful:
    print(">Saving deduped and filtered conversations...<")
    dataset_dumper(final_conversations)
elif script_mode == 2:
    print(">Saving deduped conversations...<")
    dataset_dumper(final_conversations)
elif script_mode == 3:
    print(">Saving filtered conversations...<")
    dataset_dumper(final_conversations)

if print_final_stats and not script_mode == 4:
    print(">Calculating statistics...<")
    total_tokens = 0
    total_words = 0
    total_convos = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        for i in range(0, len(final_conversations), pre_process_batch_size):
            batch = final_conversations[i:i + pre_process_batch_size]
            futures.append(executor.submit(calculate_metrics, batch))
        
        for future in tqdm(futures, desc=">Calculating conversation statistics", leave=False):
            tokens, words, convos = future.result()
            total_tokens += tokens
            total_words += words
            total_convos += convos
    
    avg_tokens_per_convo = total_tokens / total_convos if total_convos > 0 else 0
    avg_words_per_convo = total_words / total_convos if total_convos > 0 else 0
    
    print(f" Total tokens in the final dataset: {total_tokens}")
    print(f"Total words in the final dataset: {total_words}")
    print(f"Total conversations in the final dataset: {total_convos}")
    print(f"Average tokens per conversation: {avg_tokens_per_convo}")
    print(f"Average words per conversation: {avg_words_per_convo}")

print("Done!")