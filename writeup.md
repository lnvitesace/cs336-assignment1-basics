unicode1:
  a. '\x00'
  b. The `__repr__()` prints "'\\x00'", but `print` function prints nothing
  c. When occured alone, it's a string '\x00', but when in `print` function, it's just been ignored

unicode2:
  a. UTF8 is enough to represent most characters, but UTF16 and 32 encodings are too long
  b. A character can be represented as several bytes. For example, "你好" will cause UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
  c. `b'\xe4\xe5'`. `\xe4` must be followed by two bytes.

BPE Training on TinyStories:
  a. It takes 288 seconds on my M1 macbook Air, the longest token is b' accomplishment'.
  b. The max function
