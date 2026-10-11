import '../theme';
import servingCss from './serving.css?inline';
import uiCss from './ui.css?inline';

const styleId = 'vf-ui-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = uiCss + servingCss;
  document.head.append(style);
}

export { mountSteps } from './steps';
export { renderFigures } from './figures';
export { figureNames } from './figure-spec';
export {
  mountPanels,
  type PanelItem,
  type PanelsOptions,
  type LossOptions,
  type SgdOptions,
  type SequenceOptions,
  type PageSequence,
  type MatrixEntry,
  type MatrixSource,
  type MatrixHighlight,
  type TokenColumn,
} from './panels';
export { mountSections, type SectionItem, type SectionRow, type SectionBlock, type SectionsOptions, type JumpButton, type RopeSpec, type RopeCompareSpec } from './sections';
export { mountStepPanel, type StepPanelOptions, type StepPanelStep } from './step-panel';
export { mountTimeline, type TimelineSpec, type TimelineSource, type TimelineItem, type TimelineLane, type TimelineKind } from './timeline';
export type { Mounted } from './mounted';
export { mountGpuDiagram, type GpuDiagram } from './gpu-diagram';
export {
  gpuUnitIds,
  layoutGpuDiagram,
  type GpuBadgeLine,
  type GpuColor,
  type GpuDiagramProps,
  type GpuFlow,
  type GpuFocus,
  type GpuKind,
  type GpuLevel,
  type GpuPacket,
  type GpuSmOptions,
  type GpuTileGroup,
  type GpuTileOwner,
  type GpuTiles,
  type GpuTopology,
} from './gpu-layout';
export { epTopology, expandedBadges, pipelineState, scheduleTiles, transferState, type GpuState } from './gpu-sim';
export { mountServing, tokenState, cellSize, type ServingViewOptions, type ServingHandle, type TokenState } from './serving';
