from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import numpy as np
import lda
import lda.datasets
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

topicsnum=30

model = lda.LDA(n_topics=topicsnum, n_iter=500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
doc_topic = model.doc_topic_

f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([1, 3, 4, 8, 9]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, topicsnum+1)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()