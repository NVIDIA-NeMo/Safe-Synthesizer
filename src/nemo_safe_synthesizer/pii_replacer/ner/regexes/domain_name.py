# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import tldextract

from ..entity import Entity
from ..predictor import ContextSpan
from ..regex import (
    Pattern,
    RegexPredictor,
    create_exact_field_matcher,
)

LABELS = [
    "hostname",
    "domainname",
    create_exact_field_matcher("domain"),
    create_exact_field_matcher("tld"),
    create_exact_field_matcher("sld"),
    create_exact_field_matcher("registrar"),
    create_exact_field_matcher("dns"),
    # create_exact_field_matcher("host"), removed for too many FPs
    re.compile(r"host.?name"),
    re.compile(r"domain.?name"),
    re.compile(r"http.?host"),
    re.compile(r"host.?http"),
]

NEG_LABELS = [
    re.compile("dns.?recursive"),
    re.compile("recursive.?dns"),
    re.compile("dns.?software"),
    re.compile("software.?dns"),
]

SPANNER = ContextSpan(pattern_list=LABELS)

MATCHER = Pattern(
    pattern=r"([a-zA-Z0-9]+(-[a-zA-Z0-9]+)*\.)+[a-zA-Z]{2,}",
    header_contexts=LABELS,
    neg_header_contexts=NEG_LABELS,
    span_contexts=SPANNER,
)

# A more aggressive matcher for fields where the only
# thing in the field could be a hostname or domain name, this
# helps get individual computer names like "user-pc", et al
#
# This checks RFC 1123 and requires context since the raw match
# could match just about any other alphanumeric string
COMPUTER_NAMES = ["workstation"]
COMPUTER_LABELS = LABELS + COMPUTER_NAMES
ISOLATED_MATCHER = Pattern(
    pattern=r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",  # noqa
    header_contexts=COMPUTER_LABELS,
    span_contexts=ContextSpan(pattern_list=COMPUTER_LABELS),
    neg_header_contexts=NEG_LABELS,
    ignore_raw_score=True,
)


class DomainName(RegexPredictor):
    """Domain name regex pattern matcher."""

    tld_extract: tldextract.TLDExtract

    def __init__(self):
        entity = Entity.DOMAIN_NAME
        self.tld_extract = tldextract.TLDExtract(suffix_list_urls=None)
        super().__init__(name="domain_name", entity=entity, patterns=[MATCHER, ISOLATED_MATCHER])

    def validate_match(self, in_text: str, _) -> bool:
        result = self.tld_extract(in_text)
        return result.fqdn != ""


class Hostname(RegexPredictor):
    """Hostname detection. The same pattern as domain name, however
    we do not rely on requiring an extracted fqdn from the match."""

    def __init__(self):
        entity = Entity.HOSTNAME
        super().__init__(name="hostname", entity=entity, patterns=[MATCHER, ISOLATED_MATCHER])
