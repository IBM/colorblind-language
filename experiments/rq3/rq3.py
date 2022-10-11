from typing import List
from itertools import islice, chain
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, AutoModelForMaskedLM, \
    AutoConfig, AutoModel
from torch.nn import functional as F
import torch
import re
import sys
from scipy.stats import ranksums
from math import sqrt
from statistics import mean, stdev
from nltk.tokenize import word_tokenize, sent_tokenize


class WordRanker():
    def __init__(self, device="cpu"):
        model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # config = AutoConfig.from_pretrained(
        #     pretrained_model_name_or_path=model_name,
        # )
        # self.model = AutoModel.from_config(config)
        self.model = BertForMaskedLM.from_pretrained(model_name, return_dict=True).to(device)

        # model_name = "roberta-large"
        # self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        # self.model = RobertaForMaskedLM.from_pretrained(model_name, return_dict=True).to(device)

        # TODO: need to check if this work
        # if transformer_kwargs is None:
        #     transformer_kwargs = {}
        # tokenizer = PretrainedTransformerTokenizer(
        #     model_name,
        #     tokenizer_kwargs=tokenizer_kwargs,
        # )
        # self.transformer_model_for_mlm = AutoModelForMaskedLM.from_pretrained(
        #     model_name,
        #     # **transformer_kwargs,
        # )

        self.device = device

    def get_word_rank_batch(self, sentences: List[str], word: str):
        # sentences = [sentence.replace(word, self.tokenizer.mask_token) for sentence in sentences]
        # sentences = self.tokenize_and_mask(sentences, word)
        word_index = self.tokenizer.convert_tokens_to_ids(word)
        input = self.tokenizer.batch_encode_plus(sentences, return_tensors="pt", padding=True).to(self.device)
        mask_index = torch.where(input["input_ids"] == self.tokenizer.mask_token_id)[1]
        output = self.model(**input)
        logits = output.logits
        softmax = F.softmax(logits, dim=-1)
        mask_word = softmax[torch.arange(softmax.size(0)), mask_index, :]
        prediction = torch.sort(mask_word, dim=1, descending=True)[1]
        word_ranks = (prediction == word_index).nonzero(as_tuple=True)[1]
        # return word_ranks.sum().item()
        [print(rank, sentence.replace("[MASK]", word)) for rank, sentence in zip(word_ranks.tolist(), sentences)]
        return word_ranks.tolist()

    def clean_lines(self, sentences_tokens, masked_word):
        clean_lines = []
        for sentence_tokens in sentences_tokens:
            #if len(sentence_tokens) > 20: continue
            # found = False
            # sentence_tokens = word_tokenize(line.lower())
            try:
                word_index = sentence_tokens.index(masked_word)
                # word_index = sentence_tokens.index("grey")
                sent_tokens = sentence_tokens.copy()
                sent_tokens[word_index] = self.tokenizer.mask_token
                clean_lines.append(" ".join(sent_tokens))
            except ValueError:
                if masked_word == "grey":
                    try:
                        word_index = sentence_tokens.index("gray")
                        sent_tokens = sentence_tokens.copy()
                        sent_tokens[word_index] = self.tokenizer.mask_token
                        clean_lines.append(" ".join(sent_tokens))
                    except ValueError:
                        continue
                continue
        return clean_lines

    def batcher(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

def main():
    cb_input_file = sys.argv[1]
    nv_input_file = sys.argv[2]
    masked_words_file = sys.argv[3]
    filter_score = int(sys.argv[4])
    batch = int(sys.argv[5])
    device = sys.argv[6]

    word_ranker = WordRanker(device)

    with open(masked_words_file, "r") as in_file:
        masked_words = in_file.readlines()

    nv_lines = []
    count = 0
    with open(nv_input_file, "r") as in_file:
        for line in in_file.readlines():
            count += 1
            if count % 100000 == 0:
                print(f'{count} nv lines read so far')
            nv_lines.append(word_tokenize(line.lower()))
    print("Finish read and tokenize nv data lines")

    cb_lines = []
    count = 0
    with open(cb_input_file, "r") as in_file:
        for line in in_file.readlines():
            count += 1
            if count % 100000 == 0:
                print(f'{count} cb lines read so far')
            cb_lines.append(word_tokenize(line.lower()))
    print("Finish read and tokenize cb data lines")

    for masked_word in masked_words:
    # for masked_word in ["gray"]:
        masked_word = masked_word.strip()

        clean_nv_lines = word_ranker.clean_lines(nv_lines, masked_word)
        print("Finish masking nv data for ", masked_word)
        if len(clean_nv_lines) == 0:
            print(f'No nv lines found for {masked_word}. Skipping...')
            continue

        clean_cb_lines = word_ranker.clean_lines(cb_lines, masked_word)
        print("Finish masking cb data for ", masked_word)
        if len(clean_cb_lines) == 0:
            print(f'No cb lines found for {masked_word}. Skipping...')
            continue

        print(f'About to process {len(clean_nv_lines)} NV lines for {masked_word}')
        nv_total_rank = 0
        nv_rank_list = []
        nv_num_sentences = 0
        for sentences in word_ranker.batcher(clean_nv_lines, batch):
            #sentences = ["We lay on the yellow grass", "The grass is yellow", "We lay on the yellow grass", "We blow a yellow ballon", ]
            nv_rank_list += word_ranker.get_word_rank_batch(sentences, masked_word)
            nv_total_rank = sum(nv_rank_list)
            nv_num_sentences += len(sentences)
        nv_rank = nv_total_rank / len(clean_nv_lines)
        print(f'Total nv rank is {nv_total_rank}. Average nv rank is {nv_rank}')

        print(f'About to process {len(clean_cb_lines)} CB lines for {masked_word}')
        cb_rank_list = []
        scored_list = []
        for sentences in word_ranker.batcher(clean_cb_lines, batch):
            #sentences = ["We lay on the yellow grass", "The grass is yellow", "We lay on the yellow grass", "We blow a yellow ballon", ]
            tmp_ranks = word_ranker.get_word_rank_batch(sentences, masked_word)
            scored_list += zip(tmp_ranks, sentences)
            cb_rank_list += tmp_ranks
        print(f'Total cb rank is {sum(cb_rank_list)}. Average cb rank is {mean(cb_rank_list)}')

        filtered_cb_rank_list = list(filter(lambda score: score < filter_score, cb_rank_list))
        filtered_nv_rank_list = list(filter(lambda score: score < filter_score, nv_rank_list))

        # filtered_cb_rank_list = cb_rank_list
        cohens_d = (mean(filtered_cb_rank_list) - mean(filtered_nv_rank_list)) / (
            sqrt((stdev(filtered_cb_rank_list) ** 2 + stdev(filtered_nv_rank_list) ** 2) / 2))

        print('Word results:', masked_word,
              len(clean_cb_lines), len(cb_rank_list), len(filtered_cb_rank_list),
              f'{mean(cb_rank_list):.2f}', f'{mean(filtered_cb_rank_list):.2f}', f'{stdev(filtered_cb_rank_list):.2f}',
              len(clean_nv_lines), len(nv_rank_list), len(filtered_nv_rank_list),
              f'{mean(nv_rank_list):.2f}', f'{mean(filtered_nv_rank_list):.2f}', f'{stdev(filtered_nv_rank_list):.2f}',
              f'{ranksums(filtered_cb_rank_list, filtered_nv_rank_list)[1]:.8f}', f'{cohens_d}')


if __name__ == "__main__":
    main()
