<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

Personally identifiable information (PII) is information that can identify a
person, such as a name, email address, phone number, or ID number. Before
training, Safe Synthesizer replaces the PII in your data with realistic fake
values. The model then learns the shape of your data -- which columns hold
names, how emails are written, how notes mention people -- without ever seeing
the real values.

PII replacement is on by default.

## What It Looks Like

Here is one record before and after replacement:

| Column | Before | After |
|--------|--------|-------|
| Full name | Ada Lovelace | Grace Hopper |
| Email | ada.lovelace@example.com | grace.hopper@example.com |
| Sex | F | F |
| Date of birth | 1990-04-05 | 1990-04-19 |
| Notes | Ada called about her refill. | Grace called about her refill. |

A few things to notice:

- The fake values stay consistent. The email is built from the new name, and
  the "Ada" in the notes becomes "Grace", the same as the name column.
- The fake name fits the rest of the record. Because the record says "F", the
  new name is one typically given to women.
- Dates of birth move by a few days or weeks rather than being replaced
  outright, so ages stay realistic.
- Columns that describe a person but do not identify them on their own, such as
  sex, are kept as they are.

## How It Works

Replacement happens in three steps.

1. Find the PII. An AI model reads each column's name, some simple statistics,
   and a handful of sample values, and decides which columns hold PII, what
   kind of PII each holds, and how columns relate (for example, that an email
   belongs to the person named in another column). The result is a replacement
   plan: a short, readable file you can review and edit.
2. Replace structured columns. For every column in the plan, Safe Synthesizer
   generates a realistic new value, usually in the same format as the
   original. Names and
   addresses come from Nemotron-Personas, NVIDIA's datasets of synthetic
   people that follow real population statistics, or from the open-source
   Faker library.
3. Replace PII inside free text. Columns such as notes or comments are scanned
   for names, phone numbers, email addresses, ID numbers, and other PII, and
   only those words are replaced. The rest of the text is unchanged.

The same original value gets the same fake value everywhere within a record
(or within a group of records, such as all visits by one patient), so
relationships in your data survive replacement.

Only the training data is changed. Evaluation still compares the synthetic
output with your original data.

## What Gets Replaced

Safe Synthesizer replaces:

- Names: first, middle, last, and full names
- Contact details: email addresses, phone numbers, and street addresses
- Identifiers: national ID numbers (such as US Social Security numbers),
  payment card numbers, customer and patient IDs, and API keys
- Network addresses: IPv4 and IPv6 addresses
- Dates of birth

It recognizes, but keeps, information that is useful for analysis and rarely
identifies someone by itself: city, state, postal code, country, organization,
gender, race or ethnicity, and dates other than birth dates.

## Where Your Data Goes

The AI model that builds the plan runs on your own GPU by default, so your data
never leaves your machine. This needs a GPU with 80 GB of memory, such as an
NVIDIA A100 or H100.

Without such a GPU, you have two options. You can point Safe Synthesizer at a
hosted model service, in which case column samples are sent to that service, so
use one that is approved to receive your data. See
[Environment Variables -- Where planning requests go](../user-guide/environment.md#where-planning-requests-go).
Or you can write the plan yourself, which needs no AI model at all. See
[Customizing PII Replacement](../user-guide/customize-pii-replacement.md#inline-plan).

## Reviewing the Plan

You can create the replacement plan on its own, read it, correct it, and then
reuse it for every later run. This is the best way to check that the right
columns are being replaced before you train. See the
[Review the PII Replacement Plan](../tutorials/pii-replacement-review.ipynb)
tutorial for a walkthrough.

## Limitations

- Safe Synthesizer replaces only what it is confident is PII. Replacing values
  that are not PII would distort the data your model learns from, so some PII,
  especially in free text, can be missed.
- Free-text detection works best in English. The detection model also covers
  French, Spanish, German, Italian, Portuguese, and Dutch, but finds little
  PII in Chinese, Japanese, Korean, or Thai text.
- Nicknames and other informal references to a person are not linked to the
  person's real name.
- When a street address column holds a full address, Safe Synthesizer usually
  replaces only the street and keeps the city, state, and postal code.
  Combined with other columns, a postal code can help identify someone.

Replacement is not the only protection. The synthetic output is made of new
records written by the trained model, and the evaluation report's
[PII replay](evaluation.md#pii-replay) check shows any original PII that
still appears in the output.

## Next Steps

- [Review the PII Replacement Plan](../tutorials/pii-replacement-review.ipynb) --
  step-by-step tutorial
- [Customizing PII Replacement](../user-guide/customize-pii-replacement.md) --
  edit the plan, choose where fake values come from, and tune free-text
  detection
- [Configuration Reference -- Replacing PII](../user-guide/configuration.md#replacing-pii) --
  every setting, including how to turn replacement off
