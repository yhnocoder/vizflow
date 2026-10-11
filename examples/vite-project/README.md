# vizflow 的 Vite 示例项目

这个目录是一个独立的 Vite + TypeScript 项目，用构建工具的方式引用 `@yhnocoder/vizflow`，内容与 `examples/lm-head-nll.html` 相同。页面用 `<vizflow-page>` 元素写成：`index.html` 里只放这个元素，`src/main.ts` 建图后把计算图、步骤、面板和场景设为它的属性。它依赖仓库根目录的包（`file:../..`），`.npmrc` 里的 `install-links=true` 让 npm 把包复制进 `node_modules`。npm 默认为 `file:` 依赖建立符号链接，这时 vizflow 从仓库根目录的 `node_modules` 解析 three.js，页面里会有两份。复制以后 vizflow 和页面引用同一份 three.js、Solid 和 KaTeX，与从 npm 安装时的情况一致。

先在仓库根目录构建库：

```sh
npm install
npm run build
```

然后在这个目录安装依赖并运行：

```sh
npm install
npm run dev
```

开发服务器的地址以命令输出为准。开发模式下计算图默认打开有限差分检查。

构建并预览生产版本：

```sh
npm run build
npm run preview
```

`npm run build` 先运行 `tsc --noEmit` 检查类型，再运行 `vite build`，产物输出到 `dist/`。生产构建里有限差分检查默认关闭。重新构建根目录的库以后，npm 不会自动更新 `node_modules` 里的副本，要先删除它再安装：

```sh
rm -rf node_modules/@yhnocoder/vizflow
npm install
```
