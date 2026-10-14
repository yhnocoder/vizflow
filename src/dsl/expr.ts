export type IndexItem = { k: 'at'; value: Ast } | { k: 'slice'; start?: Ast; stop?: Ast; step?: Ast } | { k: 'ellipsis' };

export type Ast =
  | { k: 'num'; v: number; pos: number }
  | { k: 'str'; v: string; pos: number }
  | { k: 'bool'; v: boolean; pos: number }
  | { k: 'none'; pos: number }
  | { k: 'name'; name: string; pos: number }
  | { k: 'member'; obj: Ast; name: string; pos: number }
  | { k: 'index'; obj: Ast; items: IndexItem[]; pos: number }
  | { k: 'call'; fn: Ast; args: Ast[]; kwargs: [string, Ast][]; pos: number }
  | { k: 'list'; items: Ast[]; pos: number }
  | { k: 'neg'; arg: Ast; pos: number }
  | { k: 'invert'; arg: Ast; pos: number }
  | { k: 'not'; arg: Ast; pos: number }
  | { k: 'logic'; op: 'and' | 'or'; a: Ast; b: Ast; pos: number }
  | { k: 'bin'; op: BinaryOp; a: Ast; b: Ast; pos: number }
  | { k: 'cmp'; op: CompareOp; a: Ast; b: Ast; pos: number }
  | { k: 'cond'; test: Ast; then: Ast; other: Ast; pos: number };

export type BinaryOp = '+' | '-' | '*' | '/' | '//' | '%' | '@' | '**';

export type CompareOp = '==' | '!=' | '<' | '<=' | '>' | '>=' | 'is' | 'is not';

const COMPARE_OPS: readonly string[] = ['==', '!=', '<', '<=', '>', '>='];

export class ParseError extends Error {
  constructor(
    message: string,
    readonly offset: number,
  ) {
    super(message);
  }
}

type Token = { t: 'num'; v: number; pos: number } | { t: 'str'; v: string; pos: number } | { t: 'id'; v: string; pos: number } | { t: 'op'; v: string; pos: number } | { t: 'end'; pos: number };

const OPERATORS = ['...', '**', '//', '==', '!=', '<=', '>=', '<', '>', '~', '+', '-', '*', '/', '%', '@', '(', ')', '[', ']', ',', '=', '.', ':'];

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    const number = /^(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?/.exec(text.slice(i));
    if (number !== null && !text.startsWith('...', i)) {
      tokens.push({ t: 'num', v: Number(number[0]), pos: i });
      i += number[0].length;
      continue;
    }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i));
    if (id !== null) {
      tokens.push({ t: 'id', v: id[0], pos: i });
      i += id[0].length;
      continue;
    }
    if (c === '"' || c === "'") {
      const close = text.indexOf(c, i + 1);
      if (close < 0) throw new ParseError('unterminated string', i);
      tokens.push({ t: 'str', v: text.slice(i + 1, close), pos: i });
      i = close + 1;
      continue;
    }
    const op = OPERATORS.find((o) => text.startsWith(o, i));
    if (op === undefined) throw new ParseError(`unexpected character '${c}'`, i);
    tokens.push({ t: 'op', v: op, pos: i });
    i += op.length;
  }
  tokens.push({ t: 'end', pos: text.length });
  return tokens;
}

class Parser {
  private at = 0;

  constructor(private readonly tokens: Token[]) {}

  private peek(): Token {
    return this.tokens[this.at];
  }

  private isOp(v: string): boolean {
    const token = this.peek();
    return token.t === 'op' && token.v === v;
  }

  private take(): Token {
    return this.tokens[this.at++];
  }

  private expect(v: string): Token {
    const token = this.peek();
    if (token.t !== 'op' || token.v !== v) throw new ParseError(`expected '${v}'`, token.pos);
    return this.take();
  }

  end(): void {
    const token = this.peek();
    if (token.t !== 'end') throw new ParseError(`unexpected '${token.v}'`, token.pos);
  }

  private isWord(v: string): boolean {
    const token = this.peek();
    return token.t === 'id' && token.v === v;
  }

  expr(): Ast {
    const value = this.disjunction();
    if (!this.isWord('if')) return value;
    const at = this.take();
    const test = this.disjunction();
    if (!this.isWord('else')) throw new ParseError("expected 'else'", this.peek().pos);
    this.take();
    return { k: 'cond', test, then: value, other: this.expr(), pos: at.pos };
  }

  private disjunction(): Ast {
    let left = this.conjunction();
    while (this.isWord('or')) {
      const op = this.take();
      left = { k: 'logic', op: 'or', a: left, b: this.conjunction(), pos: op.pos };
    }
    return left;
  }

  private conjunction(): Ast {
    let left = this.inversion();
    while (this.isWord('and')) {
      const op = this.take();
      left = { k: 'logic', op: 'and', a: left, b: this.inversion(), pos: op.pos };
    }
    return left;
  }

  private inversion(): Ast {
    if (this.isWord('not')) {
      const op = this.take();
      return { k: 'not', arg: this.inversion(), pos: op.pos };
    }
    return this.comparison();
  }

  private comparison(): Ast {
    const left = this.sum();
    const token = this.peek();
    if (this.isWord('is')) {
      this.take();
      const negated = this.isWord('not');
      if (negated) this.take();
      return { k: 'cmp', op: negated ? 'is not' : 'is', a: left, b: this.sum(), pos: token.pos };
    }
    if (token.t !== 'op' || !COMPARE_OPS.includes(token.v)) return left;
    this.take();
    const node: Ast = { k: 'cmp', op: token.v as CompareOp, a: left, b: this.sum(), pos: token.pos };
    const next = this.peek();
    if (next.t === 'op' && COMPARE_OPS.includes(next.v)) throw new ParseError('write chained comparisons as separate comparisons', next.pos);
    return node;
  }

  private sum(): Ast {
    let left = this.term();
    while (this.isOp('+') || this.isOp('-')) {
      const op = this.take() as { v: BinaryOp; pos: number };
      left = { k: 'bin', op: op.v, a: left, b: this.term(), pos: op.pos };
    }
    return left;
  }

  private term(): Ast {
    let left = this.unary();
    while (this.isOp('*') || this.isOp('/') || this.isOp('//') || this.isOp('%') || this.isOp('@')) {
      const op = this.take() as { v: BinaryOp; pos: number };
      left = { k: 'bin', op: op.v, a: left, b: this.unary(), pos: op.pos };
    }
    return left;
  }

  private unary(): Ast {
    if (this.isOp('-')) {
      const op = this.take();
      return { k: 'neg', arg: this.unary(), pos: op.pos };
    }
    if (this.isOp('~')) {
      const op = this.take();
      return { k: 'invert', arg: this.unary(), pos: op.pos };
    }
    return this.power();
  }

  private power(): Ast {
    const base = this.postfix();
    if (!this.isOp('**')) return base;
    const op = this.take();
    return { k: 'bin', op: '**', a: base, b: this.unary(), pos: op.pos };
  }

  private indexItem(): IndexItem {
    if (this.isOp('...')) {
      this.take();
      return { k: 'ellipsis' };
    }
    const start = this.isOp(':') ? undefined : this.expr();
    if (!this.isOp(':')) return { k: 'at', value: start as Ast };
    this.take();
    const bound = () => (this.isOp(',') || this.isOp(']') || this.isOp(':') ? undefined : this.expr());
    const stop = bound();
    const item: IndexItem = { k: 'slice' };
    if (start !== undefined) item.start = start;
    if (stop !== undefined) item.stop = stop;
    if (this.isOp(':')) {
      this.take();
      const step = bound();
      if (step !== undefined) item.step = step;
    }
    return item;
  }

  private postfix(): Ast {
    let node = this.atom();
    for (;;) {
      if (this.isOp('.')) {
        const dot = this.take();
        const name = this.take();
        if (name.t !== 'id') throw new ParseError("expected a name after '.'", name.pos);
        node = { k: 'member', obj: node, name: name.v, pos: dot.pos };
      } else if (this.isOp('[')) {
        const open = this.take();
        const items: IndexItem[] = [this.indexItem()];
        while (this.isOp(',')) {
          this.take();
          items.push(this.indexItem());
        }
        this.expect(']');
        node = { k: 'index', obj: node, items, pos: open.pos };
      } else if (this.isOp('(')) {
        const open = this.take();
        const args: Ast[] = [];
        const kwargs: [string, Ast][] = [];
        while (!this.isOp(')')) {
          const token = this.peek();
          const next = this.tokens[this.at + 1];
          if (token.t === 'id' && next.t === 'op' && next.v === '=') {
            this.take();
            this.take();
            kwargs.push([token.v, this.expr()]);
          } else {
            if (kwargs.length > 0) throw new ParseError('positional argument after keyword argument', token.pos);
            args.push(this.expr());
          }
          if (!this.isOp(',')) break;
          this.take();
        }
        this.expect(')');
        node = { k: 'call', fn: node, args, kwargs, pos: open.pos };
      } else {
        return node;
      }
    }
  }

  private atom(): Ast {
    const token = this.take();
    if (token.t === 'num') return { k: 'num', v: token.v, pos: token.pos };
    if (token.t === 'str') return { k: 'str', v: token.v, pos: token.pos };
    if (token.t === 'id') {
      if (token.v === 'True' || token.v === 'False') return { k: 'bool', v: token.v === 'True', pos: token.pos };
      if (token.v === 'None') return { k: 'none', pos: token.pos };
      if (token.v === 'if' || token.v === 'else') throw new ParseError(`unexpected '${token.v}'`, token.pos);
      return { k: 'name', name: token.v, pos: token.pos };
    }
    if (token.t === 'op' && token.v === '(') {
      const inner = this.expr();
      if (!this.isOp(',')) {
        this.expect(')');
        return inner;
      }
      const items: Ast[] = [inner];
      while (this.isOp(',')) {
        this.take();
        if (this.isOp(')')) break;
        items.push(this.expr());
      }
      this.expect(')');
      return { k: 'list', items, pos: token.pos };
    }
    if (token.t === 'op' && token.v === '[') {
      const items: Ast[] = [];
      while (!this.isOp(']')) {
        items.push(this.expr());
        if (!this.isOp(',')) break;
        this.take();
      }
      this.expect(']');
      return { k: 'list', items, pos: token.pos };
    }
    throw new ParseError(token.t === 'end' ? 'unexpected end of expression' : `unexpected '${token.v}'`, token.pos);
  }
}

export function parseExpr(text: string): Ast {
  const parser = new Parser(tokenize(text));
  const ast = parser.expr();
  parser.end();
  return ast;
}
