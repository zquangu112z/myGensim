# import modules & set up logging
import gensim
import logging
import pprint
from scipy import spatial
from mysentences import MySentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def get_word2vec_model(dirname):
    try:
        model = gensim.models.Word2Vec.load('./tmp/mymodel')
    except:
        sentences = MySentences(dirname)  # a memory-friendly iterator
        model = gensim.models.Word2Vec(sentences,
                                       window=5,
                                       min_count=1,
                                       workers=4)  # TODO: modify min_count
        model.save('./tmp/mymodel')  # save model for further using
    return model


def retrain(model, dirname2):
    more_sentences = MySentences(dirname2)
    model.train(more_sentences)
    return model


def main():
    model = get_word2vec_model('./input')
    print('type of model: ', type(model))

    # vector represents word 'Tamhong'
    v1 = model['Tamhong']
    print('model["Tamhong"]: ', v1)

    # vocabulary
    print('Vocabulary length 1: ', len(model.wv.vocab))
    pp = pprint.PrettyPrinter()
    # pp.pprint(model.wv.vocab)

    # find the most similar word to 'Alendronate'
    print("model.most_similar('Alendronate'): ",
          model.most_similar('Alendronate'))

    # predict a word base on the context
    print(model.predict_output_word(['Alendronate',
                                     'MG',
                                     '(cid:1)',
                                     'Tablet']))

    # The model cann't generate vector representation for
    # word isn't in vocab list
    try:
        print("ahaha", model['ahaha'])
    except:
        print("1st: not existing word 'ahaha'")

    # ---Re train with new words added---
    new_sentences = MySentences('./input2')  # new words
    model.build_vocab(new_sentences, update=True)
    model.train(new_sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    try:
        print("ahaha", model['ahaha'])
    except:
        print("2nd: not existing word 'ahaha'")

    try:
        print("Tamhong", model['Tamhong'])
    except:
        print("2nd: not existing word 'Tamhong'")

    v2 = model['Tamhong']
    print("v1 vs v2: ", 1 - spatial.distance.cosine(v1, v2))

    print('Vocabulary length 2: ', len(model.wv.vocab))

if __name__ == "__main__":
    main()
