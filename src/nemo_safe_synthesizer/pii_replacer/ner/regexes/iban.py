# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""The IBAN patterns are based on the IBAN specification here:
https://en.wikipedia.org/wiki/International_Bank_Account_Number
In addition, an IBAN example per country can be found here:
git shttps://www.xe.com/ibancalculator/countrylist
An IBAN checker is available here: https://www.iban.com/iban-checker

# See https://github.com/microsoft/presidio/blob/master/presidio-analyzer/presidio_analyzer/predefined_recognizers/iban_recognizer.py  # noqa
# under MIT license.
"""

from __future__ import annotations

import string

# Import 're2' regex engine if installed, if not- import 'regex'
try:
    import re2 as re
except ImportError:
    import regex as re

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor

IBAN_GENERIC_REGEX = r"\b[A-Z]{2}[0-9]{2}[ ]?([a-zA-Z0-9][ ]?){11,28}\b"

LABELS = ["iban", "bank", "transaction"]

LETTERS = {ord(d): str(i) for i, d in enumerate(string.digits + string.ascii_uppercase)}


# IBAN parts format
CC = "[A-Z]{2}"  # country code
CK = "[0-9]{2}[ ]?"  # checksum
EOS = "$"  # end of string

A = "[A-Z][ ]?"
A2 = "([A-Z][ ]?){2}"
A3 = "([A-Z][ ]?){3}"
A4 = "([A-Z][ ]?){4}"

C = "[a-zA-Z0-9][ ]?"
C2 = "([a-zA-Z0-9][ ]?){2}"
C3 = "([a-zA-Z0-9][ ]?){3}"
C4 = "([a-zA-Z0-9][ ]?){4}"

N = "[0-9][ ]?"
N2 = "([0-9][ ]?){2}"
N3 = "([0-9][ ]?){3}"
N4 = "([0-9][ ]?){4}"

regex_per_country = {
    # Albania (8n, 16c) ALkk bbbs sssx cccc cccc cccc cccc
    "AL": "^(AL)" + CK + N4 + N4 + C4 + C4 + C4 + C4 + EOS,
    # Andorra (8n, 12c) ADkk bbbb ssss cccc cccc cccc
    "AD": "^(AD)" + CK + N4 + N4 + C4 + C4 + C4 + EOS,
    # Austria (16n) ATkk bbbb bccc cccc cccc
    "AT": "^(AT)" + CK + N4 + N4 + N4 + N4 + EOS,
    # Azerbaijan    (4c,20n) AZkk bbbb cccc cccc cccc cccc cccc
    "AZ": "^(AZ)" + CK + C4 + N4 + N4 + N4 + N4 + N4 + EOS,
    # Bahrain   (4a,14c)    BHkk bbbb cccc cccc cccc cc
    "BH": "^(BH)" + CK + A4 + C4 + C4 + C4 + C2 + EOS,
    # Belarus (4c, 4n, 16c)   BYkk bbbb aaaa cccc cccc cccc cccc
    "BY": "^(BY)" + CK + C4 + N4 + C4 + C4 + C4 + C4 + EOS,
    # Belgium (12n)   BEkk bbbc cccc ccxx
    "BE": "^(BE)" + CK + N4 + N4 + N4 + EOS,
    # Bosnia and Herzegovina    (16n)   BAkk bbbs sscc cccc ccxx
    "BA": "^(BA)" + CK + N4 + N4 + N4 + N4 + EOS,
    # Brazil (23n,1a,1c) BRkk bbbb bbbb ssss sccc cccc ccct n
    "BR": "^(BR)" + CK + N4 + N4 + N4 + N4 + N4 + N3 + A + C,
    # Bulgaria  (4a,6n,8c)  BGkk bbbb ssss ttcc cccc cc
    "BG": "^(BG)" + CK + A4 + N4 + N + N + C2 + C4 + C2 + EOS,
    # Costa Rica    (18n)   CRkk 0bbb cccc cccc cccc cc (0 = always zero)
    "CR": "^(CR)" + CK + "[0]" + N3 + N4 + N4 + N4 + N2 + EOS,
    # Croatia   (17n)   HRkk bbbb bbbc cccc cccc c
    "HR": "^(HR)" + CK + N4 + N4 + N4 + N4 + N,
    # Cyprus    (8n,16c)    CYkk bbbs ssss cccc cccc cccc cccc
    "CY": "^(CY)" + CK + N4 + N4 + C4 + C4 + C4 + C4 + EOS,
    # Czech Republic    (20n)   CZkk bbbb ssss sscc cccc cccc
    "CZ": "^(CZ)" + CK + N4 + N4 + N4 + N4 + N4 + EOS,
    # Denmark   (14n)   DKkk bbbb cccc cccc cc
    "DK": "^(DK)" + CK + N4 + N4 + N4 + N2 + EOS,
    # Dominican Republic    (4a,20n)    DOkk bbbb cccc cccc cccc cccc cccc
    "DO": "^(DO)" + CK + A4 + N4 + N4 + N4 + N4 + N4 + EOS,
    # EAt Timor    (19n) TLkk bbbc cccc cccc cccc cxx
    "TL": "^(TL)" + CK + N4 + N4 + N4 + N4 + N3 + EOS,
    # Estonia   (16n) EEkk bbss cccc cccc cccx
    "EE": "^(EE)" + CK + N4 + N4 + N4 + N4 + EOS,
    # Faroe Islands    (14n) FOkk bbbb cccc cccc cx
    "FO": "^(FO)" + CK + N4 + N4 + N4 + N2 + EOS,
    # Finland   (14n) FIkk bbbb bbcc cccc cx
    "FI": "^(FI)" + CK + N4 + N4 + N4 + N2 + EOS,
    # France    (10n,11c,2n) FRkk bbbb bsss sscc cccc cccc cxx
    "FR": "^(FR)" + CK + N4 + N4 + N2 + C2 + C4 + C4 + C + N2 + EOS,
    # Georgia   (2c,16n)  GEkk bbcc cccc cccc cccc cc
    "GE": "^(GE)" + CK + C2 + N2 + N4 + N4 + N4 + N2 + EOS,
    # Germany   (18n) DEkk bbbb bbbb cccc cccc cc
    "DE": "^(DE)" + CK + N4 + N4 + N4 + N4 + N2 + EOS,
    # Gibraltar (4a,15c)  GIkk bbbb cccc cccc cccc ccc
    "GI": "^(GI)" + CK + A4 + C4 + C4 + C4 + C3 + EOS,
    # Greece    (7n,16c)  GRkk bbbs sssc cccc cccc cccc ccc
    "GR": "^(GR)" + CK + N4 + N3 + C + C4 + C4 + C4 + C3 + EOS,
    # Greenland     (14n) GLkk bbbb cccc cccc cc
    "GL": "^(GL)" + CK + N4 + N4 + N4 + N2 + EOS,
    # Guatemala (4c,20c)  GTkk bbbb mmtt cccc cccc cccc cccc
    "GT": "^(GT)" + CK + C4 + C4 + C4 + C4 + C4 + C4 + EOS,
    # Hungary   (24n) HUkk bbbs sssx cccc cccc cccc cccx
    "HU": "^(HU)" + CK + N4 + N4 + N4 + N4 + N4 + N4 + EOS,
    # Iceland   (22n) ISkk bbbb sscc cccc iiii iiii ii
    "IS": "^(IS)" + CK + N4 + N4 + N4 + N4 + N4 + N2 + EOS,
    # Ireland   (4c,14n)  IEkk aaaa bbbb bbcc cccc cc
    "IE": "^(IE)" + CK + C4 + N4 + N4 + N4 + N2 + EOS,
    # Israel (19n) ILkk bbbn nncc cccc cccc ccc
    "IL": "^(IL)" + CK + N4 + N4 + N4 + N4 + N3 + EOS,
    # Italy (1a,10n,12c)  ITkk xbbb bbss sssc cccc cccc ccc
    "IT": "^(IT)" + CK + A + N3 + N4 + N3 + C + C3 + C + C4 + C3 + EOS,
    # Jordan    (4a,22n)  JOkk bbbb ssss cccc cccc cccc cccc cc
    "JO": "^(JO)" + CK + A4 + N4 + N4 + N4 + N4 + N4 + N2 + EOS,
    # Kazakhstan    (3n,13c)  KZkk bbbc cccc cccc cccc
    "KZ": "^(KZ)" + CK + N3 + C + C4 + C4 + C4 + EOS,
    # Kosovo    (4n,10n,2n)   XKkk bbbb cccc cccc cccc
    "XK": "^(XK)" + CK + N4 + N4 + N4 + N4 + EOS,
    # Kuwait    (4a,22c)  KWkk bbbb cccc cccc cccc cccc cccc cc
    "KW": "^(KW)" + CK + A4 + C4 + C4 + C4 + C4 + C4 + C2 + EOS,
    # Latvia    (4a,13c)  LVkk bbbb cccc cccc cccc c
    "LV": "^(LV)" + CK + A4 + C4 + C4 + C4 + C,
    # Lebanon   (4n,20c)  LBkk bbbb cccc cccc cccc cccc cccc
    "LB": "^(LB)" + CK + N4 + C4 + C4 + C4 + C4 + C4 + EOS,
    # LiechteNtein (5n,12c)  LIkk bbbb bccc cccc cccc c
    "LI": "^(LI)" + CK + N4 + N + C3 + C4 + C4 + C,
    # Lithuania (16n) LTkk bbbb bccc cccc cccc
    "LT": "^(LT)" + CK + N4 + N4 + N4 + N4 + EOS,
    # Luxembourg    (3n,13c)  LUkk bbbc cccc cccc cccc
    "LU": "^(LU)" + CK + N3 + C + C4 + C4 + C4 + EOS,
    # Malta (4a,5n,18c)   MTkk bbbb ssss sccc cccc cccc cccc ccc
    "MT": "^(MT)" + CK + A4 + N4 + N + C3 + C4 + C4 + C4 + C3 + EOS,
    # Mauritania    (23n) MRkk bbbb bsss sscc cccc cccc cxx
    "MR": "^(MR)" + CK + N4 + N4 + N4 + N4 + N4 + N3 + EOS,
    # Mauritius (4a,19n,3a)   MUkk bbbb bbss cccc cccc cccc 000m mm
    "MU": "^(MU)" + CK + A4 + N4 + N4 + N4 + N4 + N3 + A,
    # Moldova   (2c,18c)  MDkk bbcc cccc cccc cccc cccc
    "MD": "^(MD)" + CK + C4 + C4 + C4 + C4 + C4 + EOS,
    # Monaco    (10n,11c,2n)  MCkk bbbb bsss sscc cccc cccc cxx
    "MC": "^(MC)" + CK + N4 + N4 + N2 + C2 + C4 + C4 + C + N2 + EOS,
    # Montenegro    (18n) MEkk bbbc cccc cccc cccc xx
    "ME": "^(ME)" + CK + N4 + N4 + N4 + N4 + N2 + EOS,
    # Netherlands   (4a,10n)  NLkk bbbb cccc cccc cc
    "NL": "^(NL)" + CK + A4 + N4 + N4 + N2 + EOS,
    # North Macedonia   (3n,10c,2n)   MKkk bbbc cccc cccc cxx
    "MK": "^(MK)" + CK + N3 + C + C4 + C4 + C + N2 + EOS,
    # Norway    (11n) NOkk bbbb cccc ccx
    "NO": "^(NO)" + CK + N4 + N4 + N3 + EOS,
    # Pakistan  (4c,16n)  PKkk bbbb cccc cccc cccc cccc
    "PK": "^(PK)" + CK + C4 + N4 + N4 + N4 + N4 + EOS,
    # Palestinian territories   (4c,21n)  PSkk bbbb xxxx xxxx xccc cccc cccc c
    "PS": "^(PS)" + CK + C4 + N4 + N4 + N4 + N4 + N,
    # Poland    (24n) PLkk bbbs sssx cccc cccc cccc cccc
    "PL": "^(PL)" + CK + N4 + N4 + N4 + N4 + N4 + N4 + EOS,
    # Portugal  (21n) PTkk bbbb ssss cccc cccc cccx x
    "PT": "^(PT)" + CK + N4 + N4 + N4 + N4 + N,
    # Qatar (4a,21c)  QAkk bbbb cccc cccc cccc cccc cccc c
    "QA": "^(QA)" + CK + A4 + C4 + C4 + C4 + C4 + C,
    # Romania   (4a,16c)  ROkk bbbb cccc cccc cccc cccc
    "RO": "^(RO)" + CK + A4 + C4 + C4 + C4 + C4 + EOS,
    # San Marino    (1a,10n,12c)  SMkk xbbb bbss sssc cccc cccc ccc
    "SM": "^(SM)" + CK + A + N3 + N4 + N3 + C + C4 + C4 + C3 + EOS,
    # Saudi Arabia  (2n,18c)  SAkk bbcc cccc cccc cccc cccc
    "SA": "^(SA)" + CK + N2 + C2 + C4 + C4 + C4 + C4 + EOS,
    # Serbia    (18n) RSkk bbbc cccc cccc cccc xx
    "RS": "^(RS)" + CK + N4 + N4 + N4 + N4 + N2 + EOS,
    # Slovakia  (20n) SKkk bbbb ssss sscc cccc cccc
    "SK": "^(SK)" + CK + N4 + N4 + N4 + N4 + N4 + EOS,
    # Slovenia  (15n) SIkk bbss sccc cccc cxx
    "SI": "^(SI)" + CK + N4 + N4 + N4 + N3 + EOS,
    # Spain (20n) ESkk bbbb ssss xxcc cccc cccc
    "ES": "^(ES)" + CK + N4 + N4 + N4 + N4 + N4 + EOS,
    # Sweden    (20n) SEkk bbbc cccc cccc cccc cccc
    "SE": "^(SE)" + CK + N4 + N4 + N4 + N4 + N4 + EOS,
    # Switzerland   (5n,12c)  CHkk bbbb bccc cccc cccc c
    "CH": "^(CH)" + CK + N4 + N + C3 + C4 + C4 + C,
    # Tunisia   (20n) TNkk bbss sccc cccc cccc cccc
    "TN": "^(TN)" + CK + N4 + N4 + N4 + N4 + N4 + EOS,
    # Turkey    (5n,17c)  TRkk bbbb bxcc cccc cccc cccc cc
    "TR": "^(TR)" + CK + N4 + N + C3 + C4 + C4 + C4 + C2 + EOS,
    # United Arab Emirates  (3n,16n)  AEkk bbbc cccc cccc cccc ccc
    "AE": "^(AE)" + CK + N4 + N4 + N4 + N4 + N3 + EOS,
    # United Kingdom (4a,14n) GBkk bbbb ssss sscc cccc cc
    "GB": "^(GB)" + CK + A4 + N4 + N4 + N4 + N2 + EOS,
    # Vatican City  (3n,15n)  VAkk bbbc cccc cccc cccc cc
    "VA": "^(VA)" + CK + N4 + N4 + N4 + N4 + N2 + EOS,
    # Virgin Islands, British   (4c,16n)  VGkk bbbb cccc cccc cccc cccc
    "VG": "^(VG)" + CK + C4 + N4 + N4 + N4 + N4 + EOS,
}


class IBAN(RegexPredictor):
    """
    IBAN regex pattern matcher based on
    https://docs.microsoft.com/en-us/exchange/policy-and-compliance/data-loss-prevention/sensitive-information-types?view=exchserver-2019#international-banking-account-number-iban
    and https://en.wikipedia.org/wiki/International_Bank_Account_Number

    Validation includes checking for proper country code and checksum.

    Examples:
        {"iban": "GB84INXK12376708278490"}
        {"bank": "GB11DHPG65218326145843"}
        {"bank no": "GB37KLYD86034902860487"}
    """

    def __init__(self):
        patterns = [
            Pattern(
                pattern=IBAN_GENERIC_REGEX,
                context_score=Score.HIGH,
                raw_score=Score.MED,
                header_contexts=LABELS,
            )
        ]

        super().__init__(entity=Entity.IBAN_CODE, patterns=patterns)

    def validate_match(self, in_text: str, _) -> bool:
        pattern_text = in_text.replace(" ", "")
        is_valid_checksum = IBAN.__generate_iban_check_digits(pattern_text) == pattern_text[2:4]
        # score = EntityRecognizer.MIN_SCORE
        result = False
        if is_valid_checksum:
            if IBAN.__is_valid_format(pattern_text):
                result = True
            elif IBAN.__is_valid_format(pattern_text.upper()):
                result = None
        return result

    @staticmethod
    def __number_iban(iban):
        return (iban[4:] + iban[:4]).translate(LETTERS)

    @staticmethod
    def __generate_iban_check_digits(iban):
        transformed_iban = (iban[:2] + "00" + iban[4:]).upper()
        number_iban = IBAN.__number_iban(transformed_iban)
        return "{:0>2}".format(98 - (int(number_iban) % 97))

    @staticmethod
    def __is_valid_format(iban):
        country_code = iban[:2]
        if country_code in regex_per_country:
            country_regex = regex_per_country[country_code]
            return country_regex and re.match(country_regex, iban, flags=re.DOTALL | re.MULTILINE)

        return False
