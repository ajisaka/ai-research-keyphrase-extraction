import argparse
from configparser import ConfigParser

from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import MMRPhrase
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingStanford
from swisscom_ai.research_keyphrase.util.fileIO import read_file

import json
from bottle import route, run, request, response, static_file


def extract_keyphrases(embedding_distrib, ptagger, raw_text, N, lang, beta=0.55, alias_threshold=0.7):
    """
    Method that extract a set of keyphrases

    :param embedding_distrib: An Embedding Distributor object see @EmbeddingDistributor
    :param ptagger: A Pos Tagger object see @PosTagger
    :param raw_text: A string containing the raw text to extract
    :param N: The number of keyphrases to extract
    :param lang: The language
    :param beta: beta factor for MMR (tradeoff informativness/diversity)
    :param alias_threshold: threshold to group candidates as aliases
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    tagged = ptagger.pos_tag_raw_text(raw_text)
    text_obj = InputTextObj(tagged, lang)
    return MMRPhrase(embedding_distrib, text_obj, N=N, beta=beta, alias_threshold=alias_threshold)


def load_local_embedding_distributor(lang):
    assert (lang in ['en', 'de']), "Only english 'en' and german 'de' are handled"
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    if lang == 'en':
        sent2vec_model = config_parser.get('SENT2VEC', 'model_path')
    elif lang == 'de':
        sent2vec_model = config_parser.get('SENT2VEC', 'model_path_de')

    return EmbeddingDistributorLocal(sent2vec_model)


def load_local_pos_tagger(lang):
    assert (lang in ['en', 'de', 'fr']), "Only english 'en', german 'de' and french 'fr' are handled"
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    jar_path = config_parser.get('STANFORDTAGGER', 'jar_path')
    model_directory_path = config_parser.get('STANFORDTAGGER', 'model_directory_path')
    return PosTaggingStanford(jar_path, model_directory_path, lang=lang)

@route("/embedrank", method="POST")
def result():
    raw_text = request.forms.text
    n = int(request.forms.n)
    response.content_type = 'application/json'
    return json.dumps(extract_keyphrases(embedding_distributor, pos_tagger, raw_text, n, 'en'))


if __name__ == '__main__':
    embedding_distributor = load_local_embedding_distributor('en')
    pos_tagger = load_local_pos_tagger('en')

run(host="0.0.0.0", port=8080, debug=True)
