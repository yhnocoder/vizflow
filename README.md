# vizflow

vizflow（npm 包名 `@yhnocoder/vizflow`）是一个 TypeScript 库，用来编写可交互的 3D 计算图页面，讲解大模型中运算的前向与反向。页面用声明式的 API 描述张量、运算、布局和讲解，反向传播、连线和悬停卡片由库根据运算定义自动推导。

## 安装

```sh
npm install @yhnocoder/vizflow
```

包分成七个入口：`core`、`explain`、`steps`、`ui`、`scene3d`、`formula`、`theme`。页面按包名导入需要的入口，例如：

```js
import { graph } from '@yhnocoder/vizflow/core';
import { createSteps } from '@yhnocoder/vizflow/steps';
```

只导入 `core` 的页面不会加载 three.js、Solid 和 KaTeX。

## 通过 jsDelivr 引用

不用构建工具的静态站可以通过 jsDelivr 按版本号引用 `dist/` 下的产物：

```js
import { graph } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/core.js';
import { createSteps } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/steps.js';
import { mountSteps, mountPanels } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/ui.js';
import { mountScene, board, stack } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/scene3d.js';
```

同一个页面的所有入口必须来自同一个位置和同一个版本。各入口共用内部模块，混用两个版本或两个来源时，页面里会有两份运算注册表，一份里创建的节点另一份无法识别。

也可以把 `npm run build` 生成的 `dist/` 目录复制进站点，按相对路径引用。复制时保留 `dist/fonts/`，字体文件从这个目录加载。页面要通过 http 打开，浏览器不允许从 `file://` 加载 ES module。

## 本地预览

```sh
npm install
npm run dev
```

然后在浏览器打开 `http://localhost:5173/examples/index.html`，这一页列出全部示例页面，例如 `/examples/lm-head-nll.html` 和 `/examples/softmax-ce.html`。端口以命令输出为准。开发服务器把 `@yhnocoder/vizflow/<入口>` 解析到 `src/` 下的源码，修改源码后刷新页面即可看到结果。

构建产物：

```sh
npm run build
```

产物输出到 `dist/`，类型声明在 `dist/types/`。

## 测试

```sh
sh scripts/test.sh
```

这个脚本依次运行注释检查、shellcheck（已安装时）、TypeScript 类型检查和 Vitest。Vitest 包括有限差分检查、PyTorch 对拍和构建产物检查。注释检查用 `uv` 运行。

浏览器测试用 Playwright：

```sh
npm run e2e
```

没有安装 Playwright 自带的浏览器时，用环境变量 `VIZFLOW_CHROMIUM` 指定 Chromium 可执行文件：

```sh
VIZFLOW_CHROMIUM=/path/to/chromium npm run e2e
```

## 文档

- 设计文档从 `docs/design/index.html` 开始读，写接口定义、常量和设计理由。
- 写作说明 `docs/guide.md` 写怎么用 vizflow 编写页面，包括三个完整示例的代码和常见报错。
