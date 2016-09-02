from gensim import corpora, models
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')
print('test')

model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
doc = corpus.docbyoffset(0)
topics = model[doc]
print(topics)

import matplotlib.pyplot as plt
num_topics_used = [len(model[doc]) for doc in corpus]
plt.hist(num_topics_used)