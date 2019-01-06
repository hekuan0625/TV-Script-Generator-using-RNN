import numpy as np
import helper


# load in data
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
print(text[:100])
print(type(text))

# explore the training data
view_line_range = (0, 10)
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

words = text.split(' ')
print('Roughly the number of total words: {}'.format(len(words)))
print('Roughly the number of unique words: {}'.format(len(set(words))))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

# create look-up table/dictionary for words
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_unique = set(text)
    vocab_to_int = {word : i for i, word in enumerate(word_unique)}
    int_to_vocab = {vocab_to_int[word] : word for word in word_unique}
    # return tuple
    return (vocab_to_int, int_to_vocab)

# tokenize punctuation
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    dict_punc = {'.' : '||Period||', ',' : '||Comma||', '"' : '||QuotationMark||',
                 ';' : '||Semicolon||', '!' : '||ExclamationMark||', '?' : '||QuestionMark||',
                 '(' : '||LeftParentheses||', ')' : '||RightParentheses||', '-' : '||Dash||',
                 '\n' : '||Return||'}
        
    return dict_punc

# pre-process training data and save data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

# load the processed data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()