# import modules & set up logging
import gensim
import logging
import pprint
from mysentences import MySentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main():
    model = gensim.models.Word2Vec.load('./tmp/mymodel')
    new_sentences = MySentences('./input2')
    model.build_vocab(new_sentences, update=True)
    model.train(new_sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    try:
        print("ahaha", model['ahaha'])
        # print("Tamhong", model['Tamhong'])
    except:
        print("2nd: not existing word 'ahaha'")

    try:
        print("Tamhong", model['Tamhong'])
        # print("Tamhong", model['Tamhong'])
    except:
        print("2nd: not existing word 'Tamhong'")


if __name__ == "__main__":
    main()
