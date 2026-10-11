import './ops';

export { explain, type Explainer } from './explainer';
export { registerExplain, explainOf } from './registry';
export type { FlowEdge } from './edges';
export { trailingSubscript, matmulOperands, type ElemRef, type CellRef } from './elem';
export type { NodeCard, EdgeCard, CardRow } from './cards';
export { builtinExplains } from './ops';
export { positionsOf, type Positions, type Dep } from './positions';
