/* KoELECTRA (BertTokenizer, do_lower_case=False) WordPiece in plain JS, for the in-browser backend.
   Mirrors HF tokenizers: BertNormalizer(clean_text, handle_chinese_chars, no lowercase, no accent strip),
   BertPreTokenizer (whitespace + punctuation split), WordPiece (greedy longest match, "##", 100 chars max).
   Checked against the Python tokenizer by webdemo/check_tokenizer.mjs. */
"use strict";
(function (root) {
  const isWs = (c) => c === " " || c === "\t" || c === "\n" || c === "\r" || /\p{Zs}/u.test(c);
  const isCtrl = (c) => !(c === "\t" || c === "\n" || c === "\r") && /[\p{Cc}\p{Cf}]/u.test(c);
  const isPunct = (c) => {
    const cp = c.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(c);
  };
  const isCjk = (cp) => (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
    (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
    (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);

  class WordPiece {
    constructor(vocabText, { unk = "[UNK]", maxChars = 100 } = {}) {
      this.vocab = new Map();
      vocabText.split(/\r?\n/).forEach((t, i) => { if (t !== "" || i === 0) this.vocab.set(t, i); });
      this.unkId = this.vocab.get(unk);
      this.maxChars = maxChars;
    }
    normalize(text) {
      let out = "";
      for (const c of text) {
        const cp = c.codePointAt(0);
        if (cp === 0 || cp === 0xFFFD || isCtrl(c)) continue;
        if (isWs(c)) out += " ";
        else if (isCjk(cp)) out += ` ${c} `;
        else out += c;
      }
      return out;
    }
    preTokenize(text) {
      const words = [];
      for (const chunk of text.split(" ")) {
        let cur = "";
        for (const c of chunk) {
          if (isPunct(c)) { if (cur) words.push(cur); words.push(c); cur = ""; }
          else cur += c;
        }
        if (cur) words.push(cur);
      }
      return words;
    }
    encodeWord(w) {
      const chars = [...w];
      if (chars.length > this.maxChars) return [this.unkId];
      const ids = [];
      let start = 0;
      while (start < chars.length) {
        let end = chars.length, id;
        while (start < end) {
          const sub = (start ? "##" : "") + chars.slice(start, end).join("");
          id = this.vocab.get(sub);
          if (id !== undefined) break;
          end--;
        }
        if (start === end) return [this.unkId];
        ids.push(id);
        start = end;
      }
      return ids;
    }
    encode(text) {
      return this.preTokenize(this.normalize(text)).flatMap((w) => this.encodeWord(w));
    }
  }
  root.WordPiece = WordPiece;
  if (typeof module !== "undefined") module.exports = { WordPiece };
})(typeof self !== "undefined" ? self : globalThis);
