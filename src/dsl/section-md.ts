import { DslError, parseYaml, place } from './source';

type Block = string | Record<string, unknown>;

interface TextFrame {
  kind: 'root' | 'column';
  line: number;
  blocks: Block[];
  pending: string[];
  pendingLine: number;
}

interface ColumnsFrame {
  kind: 'columns';
  line: number;
  widths: (number | string)[];
  cells: Block[][];
}

interface BodyFrame {
  kind: 'lead' | 'component';
  line: number;
  name: string;
  body: string[];
  bodyLine: number;
}

type Frame = TextFrame | ColumnsFrame | BodyFrame;

const FENCE = /^(:{3,})\s*(.*?)\s*$/;
const CODE = /^\s*(```|~~~)/;
const RULE = /^-{3,}\s*$/;

export function sectionRows(source: string, file: string): unknown[] {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const rows: unknown[] = [];
  const root: TextFrame = { kind: 'root', line: 1, blocks: [], pending: [], pendingLine: 1 };
  const stack: Frame[] = [root];
  const fail = (line: number, message: string): never => {
    throw new DslError(`${file}:${line}:1: ${message}`);
  };
  const top = () => stack[stack.length - 1];
  const keep = <T extends object>(obj: T, line: number): T => {
    place(obj, undefined, file, line);
    return obj;
  };
  const addBlock = (frame: TextFrame, block: Block, line: number) => {
    frame.blocks.push(block);
    place(frame.blocks, frame.blocks.length - 1, file, line);
  };
  const flushText = (frame: TextFrame) => {
    const text = frame.pending.join('\n').trim();
    if (text !== '') {
      const block = keep({ text }, frame.pendingLine);
      place(block, 'text', file, frame.pendingLine);
      addBlock(frame, block, frame.pendingLine);
    }
    frame.pending = [];
  };
  const endRow = () => {
    flushText(root);
    if (root.blocks.length === 0) return;
    rows.push(root.blocks);
    place(rows, rows.length - 1, file, root.line);
    root.blocks = [];
  };
  let inCode = false;

  lines.forEach((text, index) => {
    const line = index + 1;
    const frame = top();
    if (frame.kind === 'lead' || frame.kind === 'component') {
      const fence = FENCE.exec(text);
      if (fence === null || (frame.kind === 'lead' && inCode)) {
        if (frame.kind === 'lead' && CODE.test(text)) inCode = !inCode;
        if (frame.body.length === 0) frame.bodyLine = line;
        frame.body.push(text);
        return;
      }
      if (fence[2] !== '') fail(line, `close '::: ${frame.name}' before opening '::: ${fence[2]}'`);
      stack.pop();
      const parent = top() as TextFrame;
      const body = frame.body.join('\n');
      if (frame.kind === 'lead') {
        const block = keep({ lead: body.trim() }, frame.line);
        place(block, 'lead', file, frame.bodyLine);
        addBlock(parent, block, frame.line);
      } else if (body.trim() === '') {
        addBlock(parent, frame.name, frame.line);
      } else {
        const block = keep({ [frame.name]: parseYaml(body, file, frame.bodyLine) }, frame.line);
        place(block, frame.name, file, frame.bodyLine);
        addBlock(parent, block, frame.line);
      }
      parent.pendingLine = line + 1;
      return;
    }
    if (frame.kind === 'columns') {
      const fence = FENCE.exec(text);
      if (fence === null) {
        if (text.trim() !== '') fail(line, "only '::: column' blocks can go inside '::: columns'");
        return;
      }
      if (fence[2] === '') {
        stack.pop();
        if (frame.cells.length === 0) fail(frame.line, "'::: columns' needs at least one '::: column'");
        const row = keep({ columns: frame.widths, cells: frame.cells }, frame.line);
        place(row, 'columns', file, frame.line);
        place(row, 'cells', file, frame.line);
        rows.push(row);
        place(rows, rows.length - 1, file, frame.line);
        root.line = line + 1;
        root.pendingLine = line + 1;
        return;
      }
      if (fence[2] !== 'column') fail(line, `only '::: column' blocks can go inside '::: columns', got '::: ${fence[2]}'`);
      stack.push({ kind: 'column', line, blocks: [], pending: [], pendingLine: line + 1 });
      return;
    }
    const here = frame as TextFrame;
    if (inCode || CODE.test(text)) {
      if (CODE.test(text)) inCode = !inCode;
      if (here.pending.length === 0) here.pendingLine = line;
      here.pending.push(text);
      return;
    }
    if (here.kind === 'root' && RULE.test(text)) {
      endRow();
      root.line = line + 1;
      root.pendingLine = line + 1;
      return;
    }
    const fence = FENCE.exec(text);
    if (fence === null) {
      if (here.pending.length === 0) here.pendingLine = line;
      here.pending.push(text);
      return;
    }
    flushText(here);
    const [name, ...args] = fence[2].split(/\s+/);
    if (name === '') {
      if (here.kind === 'root') fail(line, "':::' closes nothing");
      stack.pop();
      const columns = top() as ColumnsFrame;
      columns.cells.push(here.blocks);
      place(columns.cells, columns.cells.length - 1, file, here.line);
      return;
    }
    if (name === 'columns') {
      if (here.kind !== 'root') fail(line, "'::: columns' can only be used at the top level");
      if (args.length === 0) fail(line, "'::: columns' needs one width per column, such as '::: columns 300 1fr'");
      endRow();
      stack.push({ kind: 'columns', line, widths: args.map((a) => (/^\d+(\.\d+)?$/.test(a) ? Number(a) : a)), cells: [] });
      return;
    }
    if (name === 'column') fail(line, "'::: column' can only be used inside '::: columns'");
    if (args.length > 0) fail(line, `'::: ${name}' takes no arguments`);
    stack.push({ kind: name === 'lead' ? 'lead' : 'component', line, name, body: [], bodyLine: line + 1 });
  });

  const open = top();
  if (open.kind !== 'root') fail(open.line, `'::: ${open.kind === 'lead' || open.kind === 'component' ? open.name : open.kind}' is not closed`);
  endRow();
  return rows;
}
