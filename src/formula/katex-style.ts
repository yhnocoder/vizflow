import katexCss from 'katex/dist/katex.min.css?raw';

const styleId = 'vf-katex-style';
const fontDirectory = 'fonts/';

if (!document.getElementById(styleId)) {
  const fontBase = new URL(fontDirectory, import.meta.url).href;
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = katexCss.replaceAll('url(fonts/', `url(${fontBase}`);
  document.head.prepend(style);
}
