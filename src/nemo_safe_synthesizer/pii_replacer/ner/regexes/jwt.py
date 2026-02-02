# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import json

from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor


class JWT(RegexPredictor):
    """JSON Web Tokens"""

    def __init__(self):
        entity = Entity.JWT

        _match_1 = Pattern(
            pattern=r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*?",
            raw_score=Score.HIGH,
        )

        super().__init__(entity=entity, patterns=[_match_1])

    # https://github.com/Yelp/detect-secrets/blob/master/detect_secrets/plugins/jwt.py
    def validate_match(self, matched_text: str, _):
        parts = matched_text.split(".")
        for idx, part in enumerate(parts):
            try:
                part = part.encode("ascii")
                # https://github.com/magical/jwt-python/blob/2fd976b41111031313107792b40d5cfd1a8baf90/jwt.py#L49
                # https://github.com/jpadilla/pyjwt/blob/3d47b0ea9e5d489f9c90ee6dde9e3d9d69244e3a/jwt/utils.py#L33
                m = len(part) % 4
                if m == 1:
                    raise TypeError("Incorrect padding")
                elif m == 2:
                    part += "==".encode("utf-8")
                elif m == 3:
                    part += "===".encode("utf-8")
                b64_decoded = base64.urlsafe_b64decode(part)
                if idx < 2:
                    _ = json.loads(b64_decoded.decode("utf-8"))
            except (TypeError, ValueError, UnicodeDecodeError):
                return False

        return True
