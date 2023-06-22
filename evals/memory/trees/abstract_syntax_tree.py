import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Your sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
tokens = nltk.word_tokenize(sentence)

# Generate a Part-of-Speech (PoS) tagged parse tree
pos_tags = nltk.pos_tag(tokens)
parse_tree = nltk.ne_chunk(pos_tags)

from nltk.draw.tree import TreeView

# assuming 'parse_tree' has been defined
#TreeView(parse_tree)._cframe.print_to_file('output.ps')
nltk.draw.tree.draw_trees(parse_tree)
# Draw the parse tree
#parse_tree.pretty_print()
