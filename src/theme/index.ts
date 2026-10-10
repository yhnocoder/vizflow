import geistMono400 from '@fontsource/geist-mono/400.css?raw';
import geistMono500 from '@fontsource/geist-mono/500.css?raw';
import geist400 from '@fontsource/geist/400.css?raw';
import geist500 from '@fontsource/geist/500.css?raw';
import geist600 from '@fontsource/geist/600.css?raw';
import sourceSerif400Italic from '@fontsource/source-serif-4/400-italic.css?raw';
import sourceSerif400 from '@fontsource/source-serif-4/400.css?raw';
import sourceSerif500 from '@fontsource/source-serif-4/500.css?raw';
import themeCss from './theme.css?inline';

const styleId = 'vf-theme-style';
const fontDirectory = 'fonts/';

if (!document.getElementById(styleId)) {
  const fontBase = new URL(fontDirectory, import.meta.url).href;
  const fontCss = [sourceSerif400, sourceSerif500, sourceSerif400Italic, geist400, geist500, geist600, geistMono400, geistMono500]
    .join('\n')
    .replaceAll('url(./files/', `url(${fontBase}`);
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `${fontCss}\n${themeCss}`;
  document.head.prepend(style);
}

export { onThemeChange, readTheme, type LinearRgb, type ThemeColors } from './colors';
