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
export { mountPanels, type PanelItem, type LossOptions, type MatrixSource, type MatrixHighlight, type TokenColumn } from './panels';
export type { Mounted } from './mounted';
