# Works in py36.6, tt 1.15.0, tensorflow-hub 0.4.0
# Look here for help with installation: https://www.tensorflow.org/hub
# Look here for help with the universal sentence encoder: 
# https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15
# 
# It's possible that this is not the most efficient code . . .
 
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

top_cutoff = 2 #29142
inds = [4, 5, 6, 9, 10, 11, 12, 13, 14]

data = pd.read_csv('airbnblala/analysisData.csv')
print(data.shape)
n = data.shape[0]

# f = open("All_embeddings.csv", 'w')
# f2 = open("All_embeddings2.csv", 'w')

s = []
for ind in inds:
  for i in range(512):
    s.append(str(ind+1) + ':' + str(i))
#f.write(','.join(s) + '\n')
#f2.write(','.join([str(ind + 1) for ind in inds]) + '\n')

def text(b, t):
    s = []
    for rownum in range(b, t):
        for i in inds:
            s.append(str(data.iloc[rownum, i]))
    print(len(s))
    return s


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

#with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):
embed = hub.Module(module_url)


#word = "Elephant"
#sentence = "I am a sentence for which I would like to get its embedding."
#paragraph = (
#    "Universal Sentence Encoder embeddings also support short paragraphs. "
#    "There is no hard limit on how long the paragraph is. Roughly, the longer "
#    "the more 'diluted' the embedding will be.")
#messages = [word, sentence, paragraph]


tf.logging.set_verbosity(tf.logging.ERROR)

#    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
start = time.time()
step = 1000

for rownum in range(0, n, step):
    bot = rownum
    top = min(n-1, rownum + step)
    print(top)
    messages = text(bot, top)
    t1 = time.time()
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = np.array(session.run(embed(messages)))
    print(message_embeddings.shape)
    mb = message_embeddings.reshape(top-bot, len(inds)*512)
    mb2 = message_embeddings.reshape(top - bot, len(inds), 512)
    f.write('\n'.join([','.join([str(f) for f in row]) for row in mb]) + '\n')
    f2.write('\n'.join([','.join([str(arr) for arr in row]) for row in mb2]) + '\n')
   i print(time.time() - t1)

bot = n-1
top = n
print(top)
messages = text(bot, top)
t1 = time.time()
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = np.array(session.run(embed(messages)))
print(message_embeddings.shape)
#mb = message_embeddings.reshape(top-bot, len(inds)*512)
mb2 = message_embeddings.reshape(len(inds), 512)
bigList = []
for arr in mb2:
    bigList += [str(val) for val in arr]



f = open("All_embeddings.csv", 'a')
f.write(','.join(bigList))
f.close()


#        row = []
#        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#            row.append(','.join([str(x) for x in message_embedding]))
#        f.write(','.join(row) + '\n')
stop = time.time()
print(stop - start)

#f.close()


