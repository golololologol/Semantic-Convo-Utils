# Semantic Conversation Utilities
A repo for any scripts that i make, that involve manipulating data in conversational datasets using semantic meaning of the conversations.\
So far, i made:
* Semantic_dataset_filterer.py
## Semantic dataset filterer
This script can deduplicate and filter datasets that are millions of conversations long. And condense them down to only a handful of the most unique conversations. Made with heavy focus on speed and optimized operations.
### To run this script:
0. Have Python installed. Open your console and install: `pip install numpy`, `pip install tqdm`, and [PyTorch](https://pytorch.org/)
1. Download the Semantic_dataset_filterer.py file from this repo.
2. Open it in any text/code editor.
3. Navigate to the middle of the script, there you will see all the settings available.
4. Ensure that your dataset is in a compatible format (see Compatible formats), and input the whole path to your dataset into `file_path` variable (Do not remove the `r` before the string!).
6. See other options there, and adjust them to your wants. Most notably: the `script_mode` and `device` variables.
7. Launch it and see if you get any errors.
### Compatible formats:
What's a script useful for, if the input format if unknown? Well, not for much, so here's what formats this script accepts:
1. (Old) .jsonl, `{"conversations": ["AAAA","aaaaaaa","AAAAA"], "reversed": false, "source": "aaa_dataset", "score": 2}`\
   `conversations`: List of turns in the conversation. If we visualize the turns, it'll look like: ["User: AAAA","AI: aaaaaaa","User: AAAAA"]
   `reversed`: Bool flag, that signifies if the first turn of the conversation is taken by the LLM.\
   `source`: String that denotes from what dataset the conversation came. You can leave it blank, and the script will fill it with the name of your input dataset.\
   `score`: If you have an external tool that scores every conversation as an integer between 1 and 10, you can populate this field with those scores. Otherwise just leave blank.
2. (New) .jsonl, `{"init": "Some system message", "conversations": ["AAAA", "aaaaaaa", "AAAAA"], "source": "aaa_dataset", "dataset_quality": 2, "synthetic_origin": false, "likely_factual": false, "tags": ["aaa"]}`\
   `init`: String for the system message.\
   `conversations`: List of turns in the conversation. If we visualize the turns, it'll look like: ["User: AAAA","AI: aaaaaaa","User: AAAAA"]\
   `source`: String that denotes from what dataset the conversation came. You can leave it blank, and the script will fill it with the name of your input dataset.\
   `dataset_quality`: Same as `score`. If you have an external tool that scores every conversation as an integer between 1 and 10, you can populate this field with those scores. Otherwise just leave blank.\
   `synthetic_origin`: Bool flag, signifies if the dataset was synthetically generated, or human-made.\
   `likely_factual`: Bool flag, signifies if the conversation is likely to be correct and factual.\
   `tags`: List of strings, you can add any tags you want here, example tag: "reversed".
### Notes:
* Dataset versions are cross-loadable, say you have a dataset in the old format, and set the script's `dataset_version` to 2 (new format), the final conversations will be saved in the new format. Same the other way.
* Embedding making supports Apple's `mps` device to utilize your Apple GPU for this.
* If you're feeling extra lazy, in theory, you can make your dataset consist only of conversation turns, without any addidtional fields. Other fields should be added automatically, if you provide atleast the `conversations` lines as lists of turns.\
  So i think you can get away with a dataset thats nothing but: `{"conversations": ["AAAA","aaaaaaa","AAAAA"]}`
* `dataset_quality`/`score` can also be used with negative numbers. The conversations that have negative score/quality will be less likely to be present in the final filtered dataset, but they still will have an impact on overall uniqueness calculations.\
  So its possibly a good idea to add to your input dataset some other big one, set the big dataset's conversations score to something like -9, and in theory it'll improve the quality of the filtering, as more data will be considered when calculating uniquness of each conversation.
#### Performance(i9 9900k, RTX 3090):
* For a 6gb dataset of 4 million conversations, with the default model and dataset chunking of 100000 conversations, this script made embeddings in 24 hours on a single RTX 3090, maximum RAM usage was 26gb. Final embeddings file ended up being 16gb in size.
* This script works with minimal overheads, from profiling it: When pre-processing text for embeddings, 97.5% of time is spent waiting on the tokenizer. When making embeddings, 98.5% of time is spent waiting on the embedding model. Only path to improving this, as i see is to use ONNX, or even better, TensorRT-converted models, but thats a whole nother level of headache to deal with.
