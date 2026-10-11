export type Ast =
  | { k: 'num'; v: number; pos: number }
  | { k: 'str'; v: string; pos: number }
  | { k: 'bool'; v: boolean; pos: number }
  | { k: 'name'; name: string; pos: number }
  | { k: 'member'; obj: Ast; name: string; pos: number }
  | { k: 'index'; obj: Ast; index: Ast; end?: Ast; pos: number }
  | { k: 'call'; fn: Ast; args: Ast[]; kwargs: [string, Ast][]; pos: number }
  | { k: 'list'; items: Ast[]; pos: number }
  | { k: 'neg'; arg: Ast; pos: number }
  | { k: 'bin'; op: BinaryOp; a: Ast; b: Ast; pos: number };

export type BinaryOp = '+' | '-' | '*' | '/' | '//' | '@' | '**';

export class ParseError extends Error {
  constructor(
    message: string,
    readonly offset: number,
  ) {
    super(message);
  }
}

type Token = { t: 'num'; v: number; pos: number } | { t: 'str'; v: string; pos: number } | { t: 'id'; v: string; pos: number } | { t: 'op'; v: string; pos: number } | { t: 'end'; pos: number };

const OPERATORS = ['**', '//', '..', '+', '-', '*', '/', '@', '(', ')', '[', ']', ',', '=', '.'];

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    const number = /^(\d+(\.\d+)?|\.\d+)([eE][-+]?\d+)?/.exec(text.slice(i));
    if (number !== null && !(c === '.' && text[i + 1] === '.')) {
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
    if (c === '"') {
      const close = text.indexOf('"', i + 1);
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

  expr(): Ast {
    let left = this.prod();
    while (this.isOp('+') || this.isOp('-')) {
      const op = this.take();
      left = { k: 'bin', op: op.t === 'op' ? (op.v as BinaryOp) : '+', a: left, b: this.prod(), pos: op.pos };
    }
    return left;
  }

  private prod(): Ast {
    let left = this.power();
    while (this.isOp('*') || this.isOp('/') || this.isOp('//') || this.isOp('@')) {
      const op = this.take() as { v: BinaryOp; pos: number };
      left = { k: 'bin', op: op.v, a: left, b: this.power(), pos: op.pos };
    }
    return left;
  }

  private power(): Ast {
    const base = this.unary();
    if (!this.isOp('**')) return base;
    const op = this.take();
    return { k: 'bin', op: '**', a: base, b: this.unary(), pos: op.pos };
  }

  private unary(): Ast {
    if (this.isOp('-')) {
      const op = this.take();
      return { k: 'neg', arg: this.postfix(), pos: op.pos };
    }
    return this.postfix();
  }

  private postfix(): Ast {
    let node = this.atom();
    for (;;) {
      if (this.isOp('.')) {
        const dot = this.take();
        const name = this.take();
        if (name.t !== 'id') throw new ParseError('expected a name after \'.\'', name.pos);
        node = { k: 'member', obj: node, name: name.v, pos: dot.pos };
      } else if (this.isOp('[')) {
        const open = this.take();
        const index = this.expr();
        if (this.isOp('..')) {
          this.take();
          node = { k: 'index', obj: node, index, end: this.expr(), pos: open.pos };
        } else {
          node = { k: 'index', obj: node, index, pos: open.pos };
        }
        this.expect(']');
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
      if (token.v === 'true' || token.v === 'false') return { k: 'bool', v: token.v === 'true', pos: token.pos };
      return { k: 'name', name: token.v, pos: token.pos };
    }
    if (token.t === 'op' && token.v === '(') {
      const inner = this.expr();
      this.expect(')');
      return inner;
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

  range(): [Ast, Ast] {
    const from = this.expr();
    this.expect('..');
    return [from, this.expr()];
  }
}

export function parseExpr(text: string): Ast {
  const parser = new Parser(tokenize(text));
  const ast = parser.expr();
  parser.end();
  return ast;
}

export function parseRange(text: string): [Ast, Ast] {
  const parser = new Parser(tokenize(text));
  const range = parser.range();
  parser.end();
  return range;
}
