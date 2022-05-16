from data_utils import get_pos_vocabulary, get_stopwords, get_supported_pos_tags
pos_vocabulary = get_pos_vocabulary()
print(len(pos_vocabulary))
stop_words = get_stopwords()
supported_pos_tags = get_supported_pos_tags()