# Third-Party Licenses

This file documents third-party code included in or adapted for nemo_safe_synthesizer.

## dp-transformers (Microsoft)

Files in `privacy/dp_transformers/` have been adapted from Microsoft's dp-transformers library.

Repository: <https://github.com/microsoft/dp-transformers>

### Adapted files

- `privacy/dp_transformers/dp_utils.py` -- from `src/dp_transformers/dp_utils.py`
- `privacy/dp_transformers/sampler.py` -- from `src/dp_transformers/sampler.py`
- `privacy/dp_transformers/privacy_args.py` -- from `src/dp_transformers/arguments.py`
- `privacy/dp_transformers/linear.py` -- from `research/fine_tune_llm_w_qlora/linear.py`

License: MIT

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
