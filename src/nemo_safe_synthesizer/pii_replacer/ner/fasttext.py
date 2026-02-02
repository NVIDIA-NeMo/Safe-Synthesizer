# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Union

from nemo_safe_synthesizer.pii_replacer.ner.models import (
    ModelManifest,
    ObjectRef,
    Visibility,
    get_cache_manager,
)

# TODO: Figure out import situation and resolve noqa: F821 exceptions through
# ner/ directory.

spacy = None
dot = None
np = None


manifest = ModelManifest(
    model="fasttext",
    version="3",
    sources=[
        ObjectRef(key="pos_neg_terms", file_name="FT_posneg_terms.pickle"),
        ObjectRef(key="ft_word_vecs", file_name="FTwordvecsnormPCA.pickle"),
        ObjectRef(key="ft_ngram_vecs", file_name="FTngramvecsPCA.pickle"),
    ],
    visibility=Visibility.INTERNAL,
)
"""
Defines FTEntityMatcher model files

Changelog:
 - v1: initial model release
 - v2: update pos_neg_terms to include state and county tags (GC-59)
 - v3: FP fixes: "capital_loss" and "capital_gain" are no longer marked as locations
"""

# Note bad strings are ones that can be buried in a token or span multiple
# tokens. Thus the reason why they're not handled in the negative term list
# Note also that terms like cross, start, end, site often occur in valid
# location headers
bad_strings = frozenset(
    {
        "http:",
        "\t",
        "pdf",
        "jpg",
        "@",
        "site-address",
        "e-mail",
        "!",
        "%",
        "?",
        "$",
        ":",
        "n / a",
        "n.a",
        "site address",
    }
)


# List to use if we ever decide to put state and country back in
# bad_strings = frozenset({
#     'http:', '\t', 'pdf', 'jpg', '@', 'cross country', 'cross-country',
#     'start state', 'start-state', 'end state', 'end-state', 'site-address',
#     'out-of-state', 'e-mail', '!', '%', '?', '$', ':', 'n / a', 'n.a',
#     'site address'
# })


num_chars = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ".",
        "\n",
        "-",
        "E",
        "+",
        "e",
        " ",
        "/",
        ":",
        ";",
        "{",
        "}",
        "(",
        ")",
    }
)


def char_check(method: Union[all, any], query: str, check_set: frozenset):
    return method(map(lambda s: s in query, check_set))


class FTEntityMatcher:
    max_score: float = 0.8
    """All fasttext predictions are assigned the max_score"""

    VEC_SIM_SCORE: str = "VEC_SIM_SCORE"
    """Default key on spacy doc where vector similarity score is stored"""

    __slots__ = (
        "nlp",
        "posterms",
        "negterms",
        "tags",
        "lastword",
        "streetsuff",
        "ft_word_vecs",
        "ft_ngram_vecs",
    )

    @classmethod
    def factory(cls, model: ModelManifest = manifest) -> FTEntityMatcher:
        model_objects = get_cache_manager().resolve(manifest)
        return cls(**model_objects)

    def __init__(self, *, pos_neg_terms: dict, ft_word_vecs: dict, ft_ngram_vecs: dict):
        if spacy is None:
            raise RuntimeError("spacy module is not installed")

        if np is None or dot is None:
            raise RuntimeError("numpy is not installed")

        self.nlp = spacy.blank("en")
        self.posterms = pos_neg_terms["pos"]
        self.negterms = pos_neg_terms["neg"]
        self.tags = pos_neg_terms["tag"]
        self.lastword = pos_neg_terms["last"]
        self.streetsuff = pos_neg_terms["streetSuff"]
        self.ft_word_vecs = ft_word_vecs
        self.ft_ngram_vecs = ft_ngram_vecs

    def compute_ngrams_bytes(self, word, min_n, max_n):
        """From fasttext"""
        utf8_word = word.encode("utf-8")
        num_bytes = len(utf8_word)
        n = 0
        _MB_MASK = 0xC0
        _MB_START = 0x80
        ngrams = []
        for i in range(num_bytes):
            ngram = []
            if utf8_word[i] & _MB_MASK == _MB_START:
                continue
            j, n = i, 1
            while j < num_bytes and n <= max_n:
                ngram.append(utf8_word[j])
                j += 1
                while j < num_bytes and (utf8_word[j] & _MB_MASK) == _MB_START:
                    ngram.append(utf8_word[j])
                    j += 1
                if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                    ngrams.append(bytes(ngram))
                n += 1
        return ngrams

    def ft_hash_bytes(self, bytez):
        """Reproduces dictionary used in fastText.

        source:
            https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc
        """
        # Runtime warnings for integer overflow are raised, this is expected
        # behavior. These warnings are suppressed.
        old_settings = np.seterr(all="ignore")
        h = np.uint32(2166136261)
        for b in bytez:
            h = h ^ np.uint32(np.int8(b))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h

    @staticmethod
    def norm(vec):
        """Normalize vector"""
        length = 1.0 * np.sqrt(np.sum(vec**2))
        if length == 0:
            return vec
        else:
            return vec / length

    def get_ft_vec(self, word):
        """
        Get FastText vector for a word.  If it's OOV, gather the
        vectors for it's ngrams and average them
        """
        vec = [0] * 100
        # If FT already knows about this word, go ahead and get it's vector
        if word in self.ft_word_vecs:
            vec = self.ft_word_vecs[word]
        else:
            # Otherwise, first compute the ngrams of the word
            encoded_ngrams = self.compute_ngrams_bytes(word, 3, 3)

            # Translate the ngrams into into an index
            ngram_hashes = [self.ft_hash_bytes(n) % 100000 for n in encoded_ngrams]
            if len(ngram_hashes) == 0:
                return vec
            # For each ngram index, get the precomputed vector embedding,
            # sum the vectors and get the average which results in a vector
            # for this OOV word
            for nh in ngram_hashes:
                vec += self.ft_ngram_vecs[nh]
            vec = self.norm(vec / len(ngram_hashes))

        return vec

    def vec_sim(self, a, b):
        """Compute the cosing similarity between two vectors"""
        vec1 = np.array(self.get_ft_vec(a))
        vec2 = np.array(self.get_ft_vec(b))
        # return dot(self.norm(vec1), self.norm(vec2))
        return dot(vec1, vec2)

    @staticmethod
    def ent_score(doc, ent) -> float:
        """Determine NERPrediction.score. FT models should all have the same max_score"""
        return FTEntityMatcher.max_score

    def __call__(self, input_text: str) -> Doc:  # noqa: F821
        """
        When this component is called in the nlp spacy pipeline it
        receives a doc, processes it, and returns it.
        """
        doc = self.nlp.make_doc(input_text)
        doc.set_extension("FT_NER", default="", force=True)
        doc.set_extension("ent_score", method=FTEntityMatcher.ent_score, force=True)
        doc.set_extension(self.VEC_SIM_SCORE, default=0, force=True)

        # First break the text into the header and value and check for
        # negative words along the way
        beforeIs = True
        header = []
        value = []
        query = ""

        for token in doc:
            query = query + " " + token.text.lower()
            if token.text == "is":
                beforeIs = False
            elif beforeIs:
                header.append(token.text.lower())
            else:
                value.append(token.text.lower())
            # Also check if this token matches to our negative list
            if token.text.lower() in self.negterms:
                return doc

        # Save start and end of the value token
        if len(header) == 1:
            start = 2
        else:
            start = len(header) + 1

        end = len(doc)

        # If no negative words found, check for some other red flags

        # See if value too long
        if len(value) > 12:
            return doc

        # See if value has only one word and it's either too long or short
        elif (len(value) == 1) and ((len(value[0]) == 1) or (len(value[0]) > 15)):
            return doc

        # Check for headers with lots ;
        elif header.count(";") > 1:
            return doc

        if char_check(any, query.lower(), bad_strings):
            return doc

        if char_check(all, query[start:].lower(), num_chars):
            return doc

        # If still no red flags found, loop through header words
        # and score FT matches to positive terms
        for word in header:
            for term in self.posterms:
                score = self.vec_sim(word, term)
                thresh = self.posterms[term]
                if score >= thresh:
                    spans = []
                    entity = Span(doc, start, end, label=self.tags[term].upper())  # noqa: F821
                    spans.append(entity)
                    doc.ents = spans
                    doc._.FT_NER = self.tags[term].upper()
                    setattr(doc._, self.VEC_SIM_SCORE, "%.2f" % (score))
                    return doc

        # Check for location indicators in last word of value
        valueLength = len(value)
        if valueLength <= 1:
            return doc
        if valueLength <= 3:
            if value[valueLength - 1] in self.lastword:
                spans = []
                entity = Span(doc, start, end, label=self.tags[value[valueLength - 1]].upper())  # noqa: F821
                spans.append(entity)
                doc.ents = spans
                doc._.FT_NER = value[valueLength - 1].upper()
                setattr(doc._, self.VEC_SIM_SCORE, 1)
                return doc

        # And now one last check to look for street suffixes
        chkIndex = len(value) - 1
        if value[chkIndex] in [".", ")"]:
            if len(value) < 3:
                return doc
            chkIndex -= 1
        if value[chkIndex] in self.streetsuff:
            spans = []
            entity = Span(doc, start, end, label="ADDRESS")  # noqa: F821
            spans.append(entity)
            doc.ents = spans
            doc._.FT_NER = "ADDRESS"
            setattr(doc._, self.VEC_SIM_SCORE, 1)
            return doc

        return doc
