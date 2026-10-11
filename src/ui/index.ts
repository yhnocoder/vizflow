import '../theme';
import uiCss from './ui.css?inline';

const styleId = 'vf-ui-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = uiCss;
  document.head.append(style);
}

export { mountSteps } from './steps';
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
export type { Mounted } from './mounted';
