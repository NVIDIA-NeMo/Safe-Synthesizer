# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global Data Catalog for Nemo Safe Synthesizer Pii Replacer Core modules

When doing data discovery, all discovered data should map
back to an Enum here
"""

from enum import Enum


# NOTE:(jm) Using the Enum base class here didn't add
# value and made it less idiomatic to interchange
# the class attrs with floats when needed
class Score:
    LOW = 0.2
    MED = 0.5
    HIGH = 0.8
    PRETTY_HIGH = 0.9  # Pretty....pretty...pretty good!
    MAX = 1.0


class Entity(Enum):
    ABA_ROUTING_NUMBER = (
        "ABA Routing Number",
        "An american bank association routing number",
    )
    AGE = ("Age", "An age measured in months or years.")
    BIRTH_DATE = ("Birth Date", "Datetime indicating when an invididual was born.")
    CREDIT_CARD_NUMBER = (
        "Credit Card Number",
        "A credit card number is 12 to 19 digits long. They are used for payment transactions globally.",
    )  # noqa
    DATE = (
        "Date",
        "A date. This includes most date formats, as well as the names of common world holidays.",
    )  # noqa
    DATETIME = (
        "DateTime",
        "A date and timestamp. This includes most date/time formats.",
    )  # noqa
    # DATE_OF_BIRTH = ("Date of birth", "A date of birth.")
    DOMAIN_NAME = (
        "Domain name",
        "A domain name as defined by the DNS standard.",
    )  # noqa
    EMAIL_ADDRESS = (
        "Email",
        "An email address identifies the mailbox that emails are sent to or from. The maximum length of the domain name is 255 characters, and the maximum length of the local-part is 64 characters.",
    )  # noqa
    ETHNICITY = ("Ethnicity", "A person’s ethnic group.")
    FACEBOOK_DATA = (
        "Facebook Data",
        "API tokens and keys used for developer access to Facebook",
    )
    FIRST_NAME = (
        "First name",
        "A first name is defined as the first part of a PERSON_NAME.",
    )  # noqa
    GENERIC_KEY = (
        "Generic Keys",
        "An arbitrary character set mapped to a key that could be an API key or token",
    )
    GITHUB_TOKEN = ("GitHub Token", "A personal GitHub token")
    GOOGLE_DATA = ("Google Data", "Misc Google IDs, Keys, and Tokens")
    GOOGLE_OLC = (
        "Google Open Location Code",
        "Geocode string for identifying an area using special system developed by Google",
    )  # noqa
    GPS_COORDINATES = (
        "GPS Coordinates",
        "A combination of latitude and longitude into a single tuple",
    )
    # GCP_CREDENTIALS = ("GCP credentials", "Google Cloud Platform service account credentials. Credentials that can be used to authenticate with Google API client libraries and service accounts.")  # noqa
    GENDER = ("Gender", "A person’s gender identity.")
    HOSTNAME = (
        "Hostname",
        "A name that resolves to a specific host or system on a network",
    )  # noqa
    IBAN_CODE = (
        "IBAN code",
        """An International Bank Account Number (IBAN) is defined as an internationally agreed-upon method for identifying bank accounts. It's defined by the International Standard of Organization (ISO) 13616:2007 standard. ISO 13616:2007 was created by the European Committee for Banking Standards (ECBS). An IBAN consists of up to 34 alphanumeric characters including elements such as a country code or account number.""",
    )  # noqa
    # ICD9_CODE = ("ICD9 code", """The International Classification of Diseases, Ninth Revision, Clinical Modification (ICD-9-CM) lexicon is used to assign diagnostic and procedure codes associated with inpatient, outpatient, and physician office use in the United States. It was created by the US National Center for Health Statistics (NCHS). The ICD-9-CM is based on the ICD-9 but provides for additional morbidity detail. It's updated annually on October 1.""")  # noqa
    # ICD10_CODE = ("ICD10 code", """Like ICD-9-CM codes, the International Classification of Diseases, Tenth Revision, Clinical Modification (ICD-10-CM) lexicon is a series of diagnostic codes published by the World Health Organization (WHO) to describe causes of morbidity and mortality.""")  # noqa
    IMEI_HARDWARE_ID = (
        "IMEI",
        """An International Mobile Equipment Identity (IMEI) hardware identifier, used to identify mobile phones.""",
    )  # noqa
    IMSI_SUBSCRIBER_ID = (
        "IMSI",
        """An International Mobile Subscriber Identity (IMSI) identifier, used to identify mobile phone subscriber identities.""",
    )  # noqa
    IP_ADDRESS = (
        "IP address",
        "An Internet Protocol (IP) address (either IPv4 or IPv6).",
    )  # noqa
    JWT = ("JWT", "A JSON Web Token")
    LAST_NAME = (
        "Last name",
        "A last name is defined as the last part of a PERSON_NAME.",
    )  # noqa
    LATITUDE = (
        "Latitude",
        "The angular distance of a place north or south of the earth's equator",
    )  # noqa
    # TRANSFORM_LATLON_1KM = ("Decimal Degree Coordinates, 1KM precision", "Lat, Lon in decimal degrees with 1 KM precision")  # noqa
    # TRANSFORM_LATITUDE_1KM = ('Transform Latitude 1KM precision', "A Latitude coordinate with 2 decimal place precision")  # noqa
    LOCATION = ("Location", "A physical address or location.")
    LONGITUDE = (
        "Longitude",
        "The angular distance of a place east or west of the meridian at Greenwich, England, or west of the standard meridian of a celestial object",
    )  # noqa
    # TRANSFORM_LONGITUDE_1KM = ("Transform Longitude 1KM precision", "A longitude coordinate with 2 decimal place precision")  # noqa
    # MAC_ADDRESS = ("MAC address", "A media access control address (MAC address), which is an identifier for a network adapter.")  # noqa
    MD5 = ("MD5", "A MD5 Hash.")  # noqa
    # PASSPORT = ("Passport", "A passport number that matches passport numbers for the following countries: Australia, Canada, China, France, Germany, Japan, Korea, Mexico, The Netherlands, Poland, Singapore, Spain, Sweden, Taiwan, United Kingdom, and the United States.")  # noqa
    ORGANIZATION_NAME = ("Organization name", "An organization name.")
    RACE = ("Racial category", "The racial category of an individual.")
    PERSON_NAME = (
        "Person name",
        "A full person name, which can include first names, middle names or initials, and last names.",
    )  # noqa
    # NORP_GROUP = ("Nationality, religious, political group", "Nationality, religious or political group")  # noqa
    PHONE_NUMBER = ("Phone number", "A telephone number.")
    PHONE_NUMBER_NAMER = (
        "Phone Number North America",
        "A telephone number valid for North America (10-digits)",
    )  # noqa
    SENDGRID_CREDENTIALS = (
        "SendGrid credentials",
        "Credentials for use with the Sendgrid API",
    )
    SEX = (
        "Sex",
        "Physical and physiological feature description. Currently categorized into male and female.",
    )  # noqa
    SHA256 = ("SHA256", "A SHA256 hash.")  # noqa
    SHA512 = ("SHA512", "A SHA512 hash.")  # noqa
    SLACK_SECRETS = (
        "Slack Secrets",
        "Tokens or private information for Slack workspaces",
    )
    STRIPE_API_KEY = (
        "Stripe API Key",
        "A public or private API key for the Stripe service",
    )
    SQUARE_API_KEY = (
        "Square API Key",
        "Various keys and tokens for the Square payment service",
    )
    SWIFT_CODE = (
        "SWIFT code",
        """A SWIFT code is the same as a Bank Identifier Code (BIC). It's a unique identification code for a particular bank. These codes are used when transferring money between banks, particularly for international wire transfers. Banks also use the codes for exchanging other messages.""",
    )  # noqa
    TIME = ("Time", "A timestamp of a specific time of day.")
    TWILIO_DATA = (
        "Twilio API data",
        "Information about Twilio API access such as SIDs or API secrets",
    )
    UUID = ("UUID", "A Universally Unique Identifier (UUID).")
    URL = ("URL", "A Uniform Resource Locator (URL).")
    US_SOCIAL_SECURITY_NUMBER = (
        "US Social Security Number",
        """A United States Social Security number (SSN) is a 9-digit number issued to US citizens, permanent residents, and temporary residents. The Social Security number has effectively become the United States national identification number.""",
    )  # noqa
    US_STATE = ("USA State", "A state in the United States of America.")
    US_ZIP_CODE = (
        "US Zip Code",
        "Postal code used by the United States Postal Service",
    )  # noqa
    # GPE = ("Geo-Political Entity", "Countries, Cities, States")
    # USER_ID = ('User ID', "A unique identifier that an individual or entity would use to identify themselves to a system.")  # noqa

    def describe(self):
        return {
            "name": self.value[0],  # pylint: disable=unsubscriptable-object
            "tag": self.name.lower(),  # pylint: disable=no-member
            "description": self.value[1],  # pylint: disable=unsubscriptable-object  # noqa
        }

    @property
    def tag(self):
        return self.name.lower()  # pylint: disable=no-member


def init_empty_counter():
    tmp = {}
    for ent in list(Entity):
        tmp[ent.tag] = 0
        tmp[ent.tag + "_data"] = []
    return tmp
