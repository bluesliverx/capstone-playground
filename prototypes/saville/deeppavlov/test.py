#!/usr/bin/env python

from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model


# Run python -m spacy download en_core_web_sm

faq = build_model(configs.faq.tfidf_logreg_en_faq, download=True)
a = faq(["I need help"])
print(a)

