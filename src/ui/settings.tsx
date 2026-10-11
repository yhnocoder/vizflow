import { For, Show, createSignal, onCleanup } from 'solid-js';
import { render } from 'solid-js/web';
import { getBackgroundImage, getTheme, onThemeChange, setBackgroundImage, setTheme, type ThemeName } from '../theme';
import { resolveElement } from './mount';
import type { Mounted } from './mounted';

interface ThemeChoice {
  name: ThemeName;
  label: string;
  note: string;
  swatches: string[];
}

const themeChoices: ThemeChoice[] = [
  { name: 'default', label: '默认', note: '暖白纸张底色，深浅色跟随系统。', swatches: ['#f5f4ed', '#faf9f5', '#141413', '#d97757', '#6a9bcc'] },
  { name: 'catppuccin-latte', label: 'Catppuccin Latte', note: 'Catppuccin 的浅色调色板，只有浅色。', swatches: ['#e6e9ef', '#eff1f5', '#4c4f69', '#fe640b', '#1e66f5'] },
  { name: 'liquid-glass', label: 'Liquid Glass', note: '卡片是会折射背景的玻璃，只有浅色。Chromium 内核的浏览器显示边缘折射，其他浏览器显示磨砂玻璃。', swatches: ['#eef1f7', '#fbfcfe', '#1d1d1f', '#f2622e', '#0a7cff'] },
];

function SettingsView() {
  const [theme, setThemeSignal] = createSignal(getTheme());
  const [preview, setPreview] = createSignal<string | undefined>();
  const [error, setError] = createSignal<string | undefined>();
  const [dragging, setDragging] = createSignal(false);
  const [unsaved, setUnsaved] = createSignal(false);
  let input!: HTMLInputElement;

  const showPreview = (blob: Blob | undefined) => {
    const previous = preview();
    if (previous !== undefined) URL.revokeObjectURL(previous);
    setPreview(blob === undefined ? undefined : URL.createObjectURL(blob));
  };
  const reload = () => {
    setThemeSignal(getTheme());
    void getBackgroundImage().then(showPreview, () => showPreview(undefined));
  };
  reload();
  onCleanup(onThemeChange(reload));
  onCleanup(() => showPreview(undefined));

  const choose = (name: ThemeName, event: MouseEvent) => {
    const origin = event.detail === 0 ? undefined : { x: event.clientX, y: event.clientY };
    setTheme(name, origin);
    setThemeSignal(name);
  };
  const useImage = async (file: File | undefined) => {
    setError(undefined);
    if (file === undefined) return;
    try {
      setUnsaved(!(await setBackgroundImage(file)));
      showPreview(file);
    } catch (problem) {
      setError((problem as Error).message);
    }
  };
  const clearImage = async () => {
    setError(undefined);
    await setBackgroundImage(null);
    setUnsaved(false);
    showPreview(undefined);
  };

  return (
    <div class="vf-settings">
      <section class="vf-settings-section">
        <h2>主题</h2>
        <div class="vf-theme-grid" role="radiogroup" aria-label="主题">
          <For each={themeChoices}>
            {(choice) => (
              <button type="button" class="vf-theme-choice" role="radio" aria-checked={theme() === choice.name} data-theme-name={choice.name} onClick={(event) => choose(choice.name, event)}>
                <span class="vf-theme-swatches">
                  <For each={choice.swatches}>{(color) => <i style={{ background: color }} />}</For>
                </span>
                <span class="vf-theme-label">{choice.label}</span>
                <span class="vf-theme-note">{choice.note}</span>
              </button>
            )}
          </For>
        </div>
      </section>
      <section class="vf-settings-section">
        <h2>背景图片</h2>
        <div
          class="vf-drop"
          classList={{ 'vf-drop-over': dragging() }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            void useImage(event.dataTransfer?.files[0]);
          }}
        >
          <Show when={preview()} fallback={<span class="vf-drop-empty">没有设置背景图片</span>}>
            {(url) => <img class="vf-drop-preview" src={url()} alt="当前背景图片" />}
          </Show>
          <div class="vf-drop-actions">
            <button type="button" class="vf-settings-button" onClick={() => input.click()}>
              选择图片
            </button>
            <Show when={preview() !== undefined}>
              <button type="button" class="vf-settings-button" onClick={() => void clearImage()}>
                移除
              </button>
            </Show>
          </div>
          <input
            ref={input}
            type="file"
            accept="image/*"
            hidden
            onChange={(event) => {
              void useImage(event.currentTarget.files?.[0]);
              event.currentTarget.value = '';
            }}
          />
        </div>
        <Show when={unsaved()}>
          <p class="vf-settings-error">浏览器不允许保存图片，刷新页面后背景会消失。</p>
        </Show>
        <Show when={error()}>{(message) => <p class="vf-settings-error">{message()}</p>}</Show>
        <p class="vf-settings-note">图片只保存在这台设备的浏览器存储（IndexedDB）里，不会上传。也可以把图片拖到上面的方框里。</p>
      </section>
    </div>
  );
}

export function mountSettings(el: string | HTMLElement): Mounted {
  const target = resolveElement(el, 'mountSettings');
  return { dispose: render(() => <SettingsView />, target) };
}
