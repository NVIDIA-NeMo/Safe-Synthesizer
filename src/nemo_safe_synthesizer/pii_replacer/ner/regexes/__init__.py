# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from ..regex import RegexPredictor
from .aba_routing_number import AbaRoutingNumber
from .age import Age
from .credit_card import CreditCardNumber
from .domain_name import DomainName, Hostname
from .email import Email
from .facebook import Facebook
from .generic_key import GenericKey
from .github import Github
from .google import Google
from .google_olc import GoogleOLC
from .iban import IBAN
from .imei import IMEI
from .ip_address import IpAddress
from .jwt import JWT
from .lat_lon import GPSCoords, Latitude, Longitude
from .md5 import MD5
from .race_ethnicity import Ethnicity, Race
from .sendgrid import SendGrid
from .sex_gender import Gender, Sex
from .sha256 import SHA256
from .sha512 import SHA512
from .slack import SlackSecrets
from .square import SquareAPIKeys
from .stripe import StripeAPIKey
from .swift import SWIFT
from .twilio import TwilioAPIKeys
from .url import URL
from .us_phone import USPhone
from .us_ssn import US_SSN
from .us_zipcode import USZipcode
from .uuid import UUID

rules: Sequence[type[RegexPredictor]] = (
    AbaRoutingNumber,
    CreditCardNumber,
    DomainName,
    Hostname,
    Email,
    IBAN,
    IMEI,
    IpAddress,
    Latitude,
    Longitude,
    GPSCoords,
    MD5,
    SHA256,
    SHA512,
    SWIFT,
    URL,
    USPhone,
    US_SSN,
    USZipcode,
    UUID,
    StripeAPIKey,
    SquareAPIKeys,
    SlackSecrets,
    TwilioAPIKeys,
    JWT,
    SendGrid,
    Gender,
    Github,
    Google,
    Facebook,
    GenericKey,
    GoogleOLC,
    Sex,
    Age,
    Race,
    Ethnicity,
)
