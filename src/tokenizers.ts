type Pair = [string, string];
type BpeRank = Map<Pair, number>;

function get_pairs(word: string[]): Pair[] {
  const pairs = new Set<Pair>();
  for (let i = 1; i < word.length; i++) {
    const pair: Pair = [word[i - 1], word[i]];
    pairs.add(pair);
  }
  return Array.from(pairs);
}

function whitespaceClean(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function ord(c: string): number {
  return c.charCodeAt(0);
}

function range(start: number, stop: number): number[] {
  return Array.from({ length: stop - start }, (_, i) => start + i);
}

function bytesToUnicode(): [number[], string[]] {
  const bs = range(ord('!'), ord('~') + 1)
    .concat(range(ord('¡'), ord('¬') + 1))
    .concat(range(ord('®'), ord('ÿ') + 1));
  const cs = bs.map((b) => String.fromCharCode(b));
  let n = 0;
  range(0, 256).forEach((b) => {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(String.fromCharCode(256 + n));
      n += 1;
    }
  });
  // In the python implementation a dict is returned: dict(zip(bs, cs))
  // The problem is, later in the tokenizer, they rely on insertion ordering.
  // Here: vocab = list(bytes_to_unicode().values())
  // In JS, numeric object keys are automatically sorted
  // So return raw arrays instead
  return [bs, cs];
}

// type BpeRank = { [key: string]: number };
export class SimpleTokenizer {
  private byte_encoder: { [key: number]: string };
  private byte_decoder: { [key: string]: number };
  private encoder: { [key: string]: number };
  private decoder: { [key: number]: string };
  private bpe_ranks: { [key: string]: number }
  private cache: { [key: string]: string };
  private pat: RegExp;

  constructor(bpe: string) {
    const [bs, cs] = bytesToUnicode();
    this.byte_encoder = Object.fromEntries(bs.map((b, i) => [b, cs[i]]));
    this.byte_decoder = Object.fromEntries(Object.entries(this.byte_encoder).map((x) => [x[1], parseInt(x[0])]));
    const merges = bpe.split('\n').slice(1, 49152 - 256 - 2 + 1).map((x) => x.split(' '));
    const vocab = cs
      .concat(cs.map((x) => x + '</w>'))
      .concat(merges.map((x) => x.join('')))
      .concat(['<|startoftext|>', '<|endoftext|>']);
    this.encoder = Object.fromEntries(vocab.map((x, i) => [x, i]));
    this.decoder = Object.fromEntries(vocab.map((x, i) => [i, x]));
    this.bpe_ranks = Object.fromEntries(merges.map((x, i) => [x.join('-'), i]))
    this.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'};
    this.pat = new RegExp("<\\|startoftext\\||\\|endoftext\\||'s|'t|'re|'ve|'m|'ll|'d|\\p{L}+|\\p{N}+|[^\\s\\p{L}\\p{N}]+", 'giu');
  }

  bpe(token: string): string {
    if (this.cache[token]) {
      return this.cache[token];
    }
    let word: string[] = token.slice(0, -1).split('').concat(token.slice(-1) + '</w>')
    let pairs: Pair[] = get_pairs(word);
    if (!pairs.length) {
      return token + '</w>';
    }
    while (true) {
      const bigram: Pair = pairs
        .map((x): [Pair, number] => [x, this.bpe_ranks[x.join('-')] || Number.POSITIVE_INFINITY])
        .reduce((a, b) => a[1] < b[1] ? a : b)[0];
      if (!this.bpe_ranks[bigram.join('-')]) {
        break;
      }
      const [first, second] = bigram;
      const new_word: string[] = [];
      let i = 0;
      while (i < word.length) {
        let j = word.indexOf(first, i);
        if (j !== -1) {
          new_word.push(...word.slice(i, j));
          i = j;
        } else {
          new_word.push(...word.slice(i));
          break;
        }
        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          new_word.push(first + second);
          i += 2;
        } else {
          new_word.push(word[i]);
          i += 1;
        }
      }
      word = new_word;
      if (word.length === 1) {
        break;
      } else {
        pairs = get_pairs(word);
      }
    }
    const res: string = word.join(' ');
    this.cache[token] = res;
    return res;
  }

  encode(text: string): number[] {
    const tokens: number[] = [];
    text = whitespaceClean(text).toLowerCase();
    for (const token of text.matchAll(this.pat)) {
      tokens.push(...this.bpe(token[0]).split(' ').map((x) => this.encoder[x]));
    }
    return tokens;
  }

  decode(tokens: number[]): string {
    return tokens
      .map((x) => this.decoder[x])
      .join('')
      .replace(/<\/w+>/g, ' ')
      .trim();
  }

  tokenize(texts: string | string[], context_length = 77, truncate = false): number[][] {
    if (typeof texts === 'string') {
      texts = [texts];
    }
    const sot_token = this.encoder['<|startoftext|>']
    const eot_token = this.encoder['<|endoftext|>']
    const all_tokens = texts.map((text) => {
      let tokens = [sot_token, ...this.encode(text), eot_token];
      if (tokens.length > context_length) {
        if (truncate) {
          tokens.splice(context_length);
          tokens[tokens.length - 1] = eot_token;
        } else {
          throw new Error(`Input context length ${tokens.length} is greater than max context length ${context_length}`);
        }
      }
      return tokens.concat(Array(context_length - tokens.length).fill(0));
    });
    return all_tokens;
  }
}