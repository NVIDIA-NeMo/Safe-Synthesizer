# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from stdnum import luhn

from ..entity import Entity
from ..predictor import ContextSpan
from ..regex import Pattern, RegexPredictor

CC_REGEX = (
    r"^(?:4[0-9]{12}(?:[0-9]{3})?|[25][1-7][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]"
    r"|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$"
)

CC_2 = r"\b((4\s*\d\s*\d\s*\d)|(5\s*[0-5]\s*\d\s*\d)|(6\s*\d\s*\d\s*\d)|(1\s*\d\s*\d\s*\d)|(3\s*\d\s*\d\s*\d))[- ]?(\d\s*\d\s*\d\s*\d(\s*\d)?)[- ]?(\d\s*\d\s*\d\s*\d(\s*\d)?)[- ]?(\d\s*\d\s*\d(\s*\d)?(\s*\d)?(\s*\d)?)?\b"  # noqa


CC_LABELS = [
    "amex",
    re.compile(r"american.?express"),
    "americanexpress",
    "visa",
    "mastercard",
    re.compile(r"master.?card"),
    "mastercards",
    re.compile(r"master.?cards"),
    re.compile(r"diner's.?club"),
    re.compile(r"diners.?club"),
    "dinersclub",
    re.compile(r"discover.?card"),
    "discovercard",
    re.compile(r"discover.?cards"),
    "jcb",
    re.compile(r"japanese.?card.?bureau"),
    re.compile(r"carte.?blanche"),
    "carteblanche",
    re.compile(r"credit.?card"),
    re.compile(r"cc.?#"),
    re.compile(r"bank.?card"),
    re.compile(r"card.?num"),
    re.compile(r"credit.?card"),
    # re.compile(r"credit"),
    re.compile(r"check.?card"),
    re.compile(r"debit.?card"),
    re.compile(r"atm.?card"),
    re.compile(r"carte.?bancaire"),
    re.compile(r"(?:numéro|numero).?de.?carte"),
    re.compile(r"nº.?de.?la.?carte"),
    re.compile(r"nº.?de.?carte"),
    "kreditkarte",
    "karte",
    "karteninhaber",
    "karteninhabers",
    "kreditkarteninhaber",
    "kreditkarteninstitut",
    "kreditkartentyp",
    "eigentümername",
    "kartennr",
    "kartennummer",
    "kreditkartennummer",
    re.compile(r"kreditkarten.?nummer"),
    re.compile(r"carta.?di.?credito"),
    re.compile(r"carta.?credito"),
    re.compile(r"n\..?carta"),
    re.compile(r"n.?carta"),
    re.compile(r"nr\..?carta"),
    re.compile(r"nr.?carta"),
    re.compile(r"numero.?carta"),
    re.compile(r"numero.?della.?carta"),
    re.compile(r"numero.?di.?carta"),
    re.compile(r"tarjeta.?credito"),
    re.compile(r"tarjeta.?de.?credito"),
    re.compile(r"tarjeta.?crédito"),
    re.compile(r"tarjeta.?de.?crédito"),
    re.compile(r"tarjeta.?de.?atm"),
    re.compile(r"tarjeta.?atm"),
    re.compile(r"tarjeta.?debito"),
    re.compile(r"tarjeta.?de.?debito"),
    re.compile(r"tarjeta.?débito"),
    re.compile(r"tarjeta.?de.?débito"),
    re.compile(r"nº.?de.?tarjeta"),
    re.compile(r"no\..?de.?tarjeta"),
    re.compile(r"no.?de.?tarjeta"),
    re.compile(r"numero.?de.?tarjeta"),
    re.compile(r"número.?de.?tarjeta"),
    re.compile(r"tarjeta.?no"),
    re.compile(r"tarjetahabiente"),
    re.compile(r"cartão.?de.?crédito"),
    re.compile(r"cartão.?de.?credito"),
    re.compile(r"cartao.?de.?crédito"),
    re.compile(r"cartao.?de.?credito"),
    re.compile(r"cartão.?de.?débito"),
    re.compile(r"cartao.?de.?débito"),
    re.compile(r"cartão.?de.?debito"),
    re.compile(r"cartao.?de.?debito"),
    re.compile(r"débito.?automático"),
    re.compile(r"debito.?automatico"),
    re.compile(r"número.?do.?cartão"),
    re.compile(r"numero.?do.?cartão"),
    re.compile(r"número.?do.?cartao"),
    re.compile(r"numero.?do.?cartao"),
    re.compile(r"número.?de.?cartão"),
    re.compile(r"numero.?de.?cartão"),
    re.compile(r"número.?de.?cartao"),
    re.compile(r"numero.?de.?cartao"),
    re.compile(r"nº.?do.?cartão"),
    re.compile(r"nº.?do.?cartao"),
    re.compile(r"nº\..?do.?cartão"),
    re.compile(r"no.?do.?cartão"),
    re.compile(r"no.?do.?cartao"),
    re.compile(r"no\..?do.?cartão"),
    re.compile(r"no\..?do.?cartao"),
]


SPANNER = ContextSpan(pattern_list=CC_LABELS)

PATTERNS = [
    Pattern(
        pattern=CC_REGEX,
        header_contexts=CC_LABELS,
        span_contexts=SPANNER,
        ignore_raw_score=True,
    ),
    Pattern(
        pattern=CC_2,
        header_contexts=CC_LABELS,
        span_contexts=SPANNER,
        ignore_raw_score=True,
    ),
]


_strip_pattern = re.compile(r"[^0-9]")


def _is_luhn_checksum_valid(data: str) -> bool:
    data = re.sub(_strip_pattern, "", data)
    return luhn.is_valid(data)


class CreditCardNumber(RegexPredictor):
    """Credit card  number regex pattern matcher."""

    def __init__(self):
        entity = Entity.CREDIT_CARD_NUMBER
        super().__init__(entity=entity, patterns=PATTERNS)

    def validate_match(self, matched_text: str, _):
        return _is_luhn_checksum_valid(matched_text)
