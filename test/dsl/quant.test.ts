import { describe, expect, it } from 'vitest';
import { type PageFile, compilePage } from '../../src/dsl';
import { parseYaml } from '../../src/dsl/source';

const X = '[[0.5, -1.25, 6.0, 1.45], [-0.3, 0.8, 1.1, -0.45]]';

function source(weight: string, body: string, head = ''): string {
  return `title: t
${head}dims: { T: 2, d: 4, b: 2 }
real: { name: Big, T: 4096, d: 7168, b: 128 }
inputs:
  x: { shape: [T, d], value: ${X} }
weights:
  W: ${weight}
values:
  W: [[0.3, -0.2, 0.9, 0.1], [-0.7, 0.4, 0.2, 0.5], [0.6, 0.8, -0.1, -0.3], [0.05, -0.9, 0.35, 0.7]]
body:
${body}
  - loss: mean(z ** 2)
`;
}

const compile = (text: string) => compilePage(parseYaml(text, 'page.yaml') as PageFile);

describe('weights with quant', () => {
  it('inserts a Quant instance and gives references to the weight the dequantized value', async () => {
    const page = await compile(source('{ shape: [d, d], quant: { block: [b, b], rule: ue8m0 } }', '  - z: x @ W.T'));
    const master = page.nodes.get('W')!;
    const xq = page.nodes.get('W_quant.xq')!;
    expect(master.kind).toBe('param');
    expect(page.nodes.get('W_quant.x')).toBe(master);
    const z = page.nodes.get('z')!;
    expect(z.inputs[1].op).toBe('transpose');
    expect(z.inputs[1].inputs[0]).toBe(xq);
    expect(xq.realShape).toEqual([7168, 7168]);
    const s = page.nodes.get('W_quant.s')!;
    expect(s.attrs).toMatchObject({ br: 2, bc: 2, fmt: 'fp8_e4m3', rule: 'ue8m0' });
    expect(s.dtype).toBe('e8m0');
    expect(s.tex).toBe('\\mathbf{s}_{W}');
    expect(xq.tex).toBe('\\hat{W}');
    expect(page.nodes.get('W_quant.e')!.grad).toBeNull();
    expect(master.grad).not.toBeNull();
    const g = xq.grad!;
    master.grad!.data.forEach((v, i) => expect(v).toBeCloseTo(g.data[i], 12));
    expect(page.modules.find((m) => m.id === 'W_quant')).toMatchObject({ title: 'W_quant', subtitle: '块缩放量化与反量化', folded: true });
  });

  it('builds the same steps, brace and folding as a hand-written Quant instance', async () => {
    const inserted = await compile(source('{ shape: [d, d], quant: { block: [b, b], folded: false } }', '  - z: x @ W.T'));
    const written = await compile(
      source(
        '{ shape: [d, d] }',
        "  - W_quant: { use: Quant, suffix: '\\of{W}', params: { br: b, bc: b }, inputs: { x: W }, outputs: { xq: Wd }, folded: false }\n  - z: x @ Wd.T",
        'import: { Quant: vizflow/nn/quant }\n',
      ),
    );
    const stepsOf = (p: typeof inserted) => p.steps.forward.map((st) => [st.title, st.label, st.module, st.text, st.backward, st.brace]);
    expect(stepsOf(inserted)).toEqual(stepsOf(written));
    const modulesOf = (p: typeof inserted) => p.modules.map((m) => [m.id, m.title, m.subtitle, m.folded, m.fixed, m.formula, m.nodes.map((n) => n.name)]);
    expect(modulesOf(inserted)).toEqual(modulesOf(written));
    expect(inserted.nodes.get('loss')!.value.data[0]).toBe(written.nodes.get('loss')!.value.data[0]);
  });

  it('lets a hand-written instance sit beside the inserted one', async () => {
    const page = await compile(
      source(
        '{ shape: [d, d], quant: { block: [b, b] } }',
        '  - x_quant: { use: Quant, beside: W_quant, suffix: x, params: { br: 1, bc: b }, inputs: { x: x }, outputs: { xq: xq } }\n  - z: xq @ W.T',
        'import: { Quant: vizflow/nn/quant }\n',
      ),
    );
    const parallel = page.tree.children.find((e) => e.kind === 'parallel');
    expect(parallel?.kind === 'parallel' && parallel.branches.map((b) => b.title)).toEqual(['W_quant', 'x_quant']);
    const scale = page.steps.forward[0];
    expect(scale.show).toEqual([page.nodes.get('W_quant.s'), page.nodes.get('x_quant.s')]);
  });

  it('rejects unknown and malformed quant fields', async () => {
    const quant = (spec: string) => compile(source(`{ shape: [d, d], quant: ${spec} }`, '  - z: x @ W.T'));
    await expect(quant('{ block: [b, b], size: 3 }')).rejects.toThrow("weights.W.quant.size: unknown key 'size'");
    await expect(quant('{ block: [b] }')).rejects.toThrow('block must be a list [rows, cols]');
    await expect(quant('{ block: [b, b], fmt: fp6 }')).rejects.toThrow('fmt must be one of fp8_e4m3, fp8_e5m2, fp4_e2m1, int8, int4');
    await expect(quant('{ block: [b, b], rule: max }')).rejects.toThrow('rule must be one of amax, ue8m0, mx, e4m3');
    await expect(quant('{ block: [b, b], folded: no }')).rejects.toThrow('folded must be true or false');
    await expect(quant('fp8')).rejects.toThrow('quant must be a mapping');
  });
});
