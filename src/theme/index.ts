import './fonts';
import themeCss from './theme.css?inline';

const styleId = 'vf-theme-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = themeCss;
  document.head.prepend(style);
}

export { getBackgroundImage, getTheme, setBackgroundImage, setTheme, themeNames, type RevealOrigin, type ThemeName } from './settings';
export { onThemeChange, readTheme, type LinearRgb, type ThemeColors } from './colors';
