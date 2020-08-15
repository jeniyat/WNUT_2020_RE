# -*- coding: utf-8 -*-
"""Class to handle the contents of a file containing unigrams."""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from collections import Counter, OrderedDict

from tqdm import tqdm

import features_config as cfg


class Unigrams(object):
    """Class to handle the contents of a file containing unigrams."""

    def __init__(self, articles, filepath=None, skip_first_n=0, max_count_words=None):
        """Initialize the unigrams list, optionally from a file.
        Args:
            filepath: Optional filepath to a file containing the unigrams in the
                form "word<tab>count<linebreak>word<tab>count...".
            skip_first_n: Number of words to skip at the start of the unigrams file.
            max_count_words: Maxmimum number of words to read from the unigrams file.
        """
        self.articles = articles
        self.word_to_rank = OrderedDict()
        self.word_to_count = OrderedDict()
        self.sum_of_counts = 0
        # check if filepath is given and that it is not empty
        if filepath is not None:
            if not os.path.isfile(filepath) or os.stat(filepath).st_size == 0:
                self.generate_unigrams(filepath)
            else:
                self.fill_from_file(filepath, skip_first_n=skip_first_n, max_count_words=max_count_words)
        else:
            self.fill_from_articles(self.articles, verbose=True, skip_first_n=skip_first_n, max_count_words=max_count_words)

    def generate_unigrams(self, filepath):

        print("Collecting unigrams...")
        self.fill_from_articles(cfg.ARTICLES_FOLDERPATH, verbose=True)
        self.write_to_file(filepath)
        print("Finished.")

    def clear(self):
        """Resets this class, empties all ranking and count dictionaries."""
        self.word_to_rank = OrderedDict()
        self.word_to_count = OrderedDict()

    def fill_from_file(self, filepath, skip_first_n=0, max_count_words=None):
        """Fills the dictionaries of this class from a file containing unigrams.
        Expected structure of the file:
            foo    120
            bar    105
            ...
        The words have to separated from the counts by a tab.
        The words have to ordered descencnding, from the most common to the least common word.

        Args:
            filepath: Optional filepath to a file containing the unigrams in the
                form "word<tab>count<linebreak>word<tab>count...".
            skip_first_n: Number of words to skip at the start of the unigrams file.
            max_count_words: Maxmimum number of words to read from the unigrams file.
        """
        skipped = 0
        added = 0
        with open(filepath, "r", encoding='utf-8') as handle:
            for line_idx, line in enumerate(handle):
                columns = line.strip().split("\t")
                if len(columns) == 2:
                    if skipped < skip_first_n:
                        skipped += 1
                    else:
                        if max_count_words is not None and added >= max_count_words:
                            break
                        else:
                            word, count = columns
                            count = int(count)
                            self.word_to_rank[word] = added + 1
                            self.word_to_count[word] = count
                            self.sum_of_counts += count
                            added += 1
                else:
                    print("[Warning] Expected 2 columns in unigrams file at line %d, " \
                          "got %d" % (line_idx, len(columns)))

    def fill_from_articles(self, articles, verbose=False, skip_first_n=0, max_count_words=20000):
        """Fills the dictionaries of this class from a corpus file.

        The corpus file is expected to contain one article/document per line.
        The corpus file may contain labels at each word, e.g. "John/PER Doe/PER did yesterday...".
        These labels will be ignored by this function (e.g. only "John" and "Doe" would be used).
        All count values and ranks will be automatically estimated.

        Note: This function is rather slow.

        Args:
            articles: Filepath to the corpus file.
            verbose: Whether to output messages during parsing.
        """
        self.fill_from_articles_labels(articles, labels=None, verbose=verbose, skip_first_n=skip_first_n, max_count_words=max_count_words)

    def fill_from_articles_labels(self, articles, labels=None, verbose=False, skip_first_n=0, max_count_words=20000):
        """Fills the dictionaries of this class from a corpus file, optionally only with words of
        specific labels (e.g. only words labeled with "PER").

        The corpus file is expected to contain one article/document per line.
        The corpus file may contain labels at each word, e.g. "John/PER Doe/PER did yesterday...".
        All count values and ranks will be automatically estimated.

        Note: This function is rather slow.

        Args:
            articles: Filepath to the corpus file.
            labels: Optionally one or more labels. If provided, only words that are annotated
                with any of these labels will be counted.
            verbose: Whether to output messages during parsing.
        """
        assert labels is None or isinstance(labels, list)
        assert labels is None or len(labels) > 0

        counts = Counter()
        self.sum_of_counts = 0

        for i, article in enumerate(tqdm(articles, desc="Loading top N Unigrams")):
            if article.status:
                # unroll tokens2d into one large 1d tokens
                tokens = [item for sublist in article.tokens2d for item in sublist]
                words = [token.word for token in tokens
                         if labels is None or token.label in labels]
                counts.update(words)

        most_common = counts.most_common()
        for i, (word, count) in enumerate(most_common):
            if i < skip_first_n:
                continue
            else:
                if max_count_words is not None and i >= max_count_words:
                    break
                else:
                    self.word_to_count[word] = count
                    self.word_to_rank[word] = i + 1
                    self.sum_of_counts += count

    def write_to_file(self, filepath):
        """Writes the contents of this unigrams object to a file.
        The file can later on be loaded with fill_from_file().
        Args:
            filepath: Filepath to the file to which to write.
        """
        with open(filepath, "w+", encoding='utf-8') as handle:
            for i, (word, count) in enumerate(self.word_to_count.items()):
                print(word, count)
                if i > 0:
                    handle.write("\n")
                handle.write(word)
                handle.write("\t")
                handle.write(str(count))

    def get_rank_of(self, word, default=-1):
        """Returns the rank of a word among all unigrams.
        The most common word has rank 1.
        Args:
            word: The word which to rank.
            default: A default value to return if the word was not found among the unigrams.
        Returns:
            integer or default value (-1).
        """
        if word in self.word_to_rank:
            return self.word_to_rank[word]
        else:
            return default

    def get_count_of(self, word, default=-1):
        """Returns the count of a word among all unigrams.
        Args:
            word: The word which to rank.
            default: A default value to return if the word was not found among the unigrams.
        Returns:
            integer or default value (-1).
        """
        if word in self.word_to_count:
            return self.word_to_count[word]
        else:
            return default

    def get_frequency_of(self, word, default=None):
        """Returns the frequency of a word among all unigrams.
        The frequency is calculated by count(word)/count(all words)
        Args:
            word: The word which to rank.
            default: A default value to return if the word was not found among the unigrams.
        Returns:
            double or default value (None).
        """
        count = self.get_count_of(word, None)
        if count is not None:
            return count / self.sum_of_counts
        else:
            return default
