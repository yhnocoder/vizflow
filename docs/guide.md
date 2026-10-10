# vizflow 写作说明

这份说明写给用 vizflow 编写页面的人，包括作者本人和帮作者写页面的 Claude。读完以后应当能从零写出一个页面：建立计算图，写前向讲解，让库生成反向步骤，挂上步骤条、面板和 3D 场景。

接口的完整定义与设计理由在设计文档 `docs/design/pages/vizflow.html`。这份说明只写怎么用。三个完整示例页面在仓库里：

- `examples/lm-head-nll.html`：lm_head 与 NLL Loss，带 3D 场景。
- `examples/softmax-ce.html`：Softmax 与交叉熵，单样本与批量两种模式，带 3D 场景。
- `examples/widget-softmax.html`：只用面板和步骤条的 softmax 小部件，不加载 3D。

## 1. 安装与引用

### 1.1 七个入口

npm 包名是 `@yhnocoder/vizflow`。包按用途拆成七个入口，页面只引入需要的几个：

| 入口 | 内容 | 什么时候引入 |
|---|---|---|
| `@yhnocoder/vizflow/core` | 张量、计算图、运算、有限差分检查 | 每个页面都要 |
| `@yhnocoder/vizflow/steps` | 步骤状态与正文模板 | 有分步讲解时 |
| `@yhnocoder/vizflow/ui` | 步骤条、讲解正文、面板 | 有步骤条或面板时 |
| `@yhnocoder/vizflow/scene3d` | 3D 场景与布局模板 | 有 3D 场景时 |
| `@yhnocoder/vizflow/formula` | 公式渲染（KaTeX） | 页面自己的文字里要渲染公式时（`renderText`） |
| `@yhnocoder/vizflow/theme` | 颜色变量、字体、布局类 | 只用 core 或 steps 又需要样式时 |
| `@yhnocoder/vizflow/explain` | 连线、局部导数、卡片内容 | 给自定义运算注册讲解时 |

`ui` 与 `scene3d` 自动引入 `theme` 和 `formula`，用到它们的页面不需要再引入 `theme`。`steps` 自动引入 `explain`。只引入 `core` 时不会加载 three.js、Solid 和 KaTeX。

### 1.2 在本仓库里写页面与预览

本仓库的示例页面放在 `examples/`，用 Vite 的开发服务器预览：

```sh
npm install
npm run dev
```

然后在浏览器打开 `http://localhost:5173/examples/lm-head-nll.html`（端口以命令输出为准）。开发服务器把 `@yhnocoder/vizflow/<入口>` 解析到 `src/` 下的源码，所以页面里直接写包名导入，修改源码后刷新页面即可。新页面放在 `examples/` 下，例如 `examples/my-page.html`，地址是 `/examples/my-page.html`。

### 1.3 在其他项目里使用

用 npm 和构建工具（例如 Vite）时：

```sh
npm install @yhnocoder/vizflow
```

页面里按包名导入，构建工具按 `package.json` 的 `exports` 找到产物：

```js
import { graph } from '@yhnocoder/vizflow/core';
```

不用构建工具的静态站，可以通过 jsDelivr 按版本号引用，也可以把 `dist/` 目录（`npm run build` 的产物）复制进站点后按相对路径引用：

```js
import { graph } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/core.js';
import { createSteps } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/steps.js';
import { mountSteps, mountPanels } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/ui.js';
import { mountScene, board, stack, node } from 'https://cdn.jsdelivr.net/npm/@yhnocoder/vizflow@0.1.0/dist/scene3d.js';
```

同一个页面的所有入口必须来自同一个位置和同一个版本。各入口共用内部模块（运算注册表、`Node` 类），混用两个来源会得到两份注册表，面板无法识别另一份里创建的节点。

### 1.4 必须通过 http 打开

页面用 `<script type="module">` 加载，浏览器不允许从 `file://` 加载 ES module。直接双击 HTML 文件打开时页面是空白的，控制台报跨域错误。本地预览用 `npm run dev`，或者在页面所在目录启动任意静态 http server（例如 `python3 -m http.server`），再用 `http://` 地址打开。

### 1.5 页面骨架

页面的 HTML 只放容器，所有内容由脚本挂载：

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>页面标题</title>
<link rel="icon" href="data:,">
<style>
#stage { height: var(--vf-stage-height); min-height: 400px; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
</style>
</head>
<body>
<main class="vf-page">
  <h1>页面标题</h1>
  <div class="vf-split">
    <div>
      <div id="stage"></div>
      <div id="steps"></div>
    </div>
    <aside id="side"></aside>
  </div>
</main>
<script type="module">
import { graph } from '@yhnocoder/vizflow/core';
import { createSteps } from '@yhnocoder/vizflow/steps';
import { mountSteps, mountPanels } from '@yhnocoder/vizflow/ui';
import { mountScene, stack } from '@yhnocoder/vizflow/scene3d';

const g = graph({ ... });
// 建图、写步骤、挂载，见下文
</script>
</body>
</html>
```

脚本按三段写：建图（第 2 节）、写讲解（第 3、4 节）、挂载界面（第 5、6 节）。3D 场景铺满容器，容器必须有高度，上面的 `#stage` 用 theme 提供的变量 `--vf-stage-height`。

`<link rel="icon" href="data:,">` 让浏览器不去请求 `/favicon.ico`。没有这一行时，服务器上没有 favicon 会在控制台留下一条 404 错误，断言「控制台无错误」的测试会失败。

不需要 3D 场景的小部件去掉 `#stage`，把步骤条和面板竖排放进 `.vf-stack`：

```html
<main class="vf-page">
  <h1>页面标题</h1>
  <div class="vf-stack">
    <div id="steps"></div>
    <div id="side"></div>
  </div>
</main>
```

这时脚本不引入 `scene3d`，页面不会加载 three.js。

## 2. 建图

### 2.1 创建计算图

```js
import { graph } from '@yhnocoder/vizflow/core';

const g = graph({
  dims: { d: 4, V: 5 },
  real: { d: 4096, V: 151936, name: 'Qwen3-8B' },
  check: true,
});
```

三个选项都可以省略：

- `dims`：命名维度。之后写 shape 时可以用名字代替数字，例如 `[1, 'd']`。用到没有定义的名字时抛错。
- `real`：真实模型的尺寸。给出后，层名卡片多显示一行「真实模型」，例如 `Qwen3-8B：(1, 151936)`。
- `check`：打开有限差分检查。每创建一个运算节点、每次重算之后，都对每个运算检查 `backward` 与数值导数是否一致，不一致时抛出 `gradient check failed: ...`。写页面和调试自定义运算时打开；内置运算已经在测试中检查过，发布的页面可以关掉。

### 2.2 叶子节点：input、param、label

```js
const h = g.input('h', [1, 'd'], [0.8, -1.1, 0.5, 1.4]);
const W = g.param('W', ['V', 'd'], [
  [ 0.9, -0.6,  0.3,  0.6],
  [ 0.2, -0.8,  0.6,  0.2],
  [-0.4,  0.2,  0.9,  0.3],
  [ 0.5,  0.7, -0.6, -0.2],
  [-0.7,  0.4, -0.3, -0.5],
], { tex: 'W', elem: 'W' });
const t = g.label('t', [0], { tokens: [' mat', ' floor', ' sofa', ' dog', ' sat'] });
```

- `g.input(name, shape, values, options?)` 与 `g.param(...)` 创建数值叶子。`values` 可以是扁平数组，也可以是嵌套数组，元素个数必须等于 shape 的元素个数。两者在计算上没有区别，都有梯度，都可以 `set`，都可以放进损失面板的 `sgd.params`、`random.nodes` 和滑块。区分它们是为了表达含义：模型权重用 `param`，输入数据和 logits 用 `input`。只有一个 logits 输入的小部件把它放进 `sgd.params`，「梯度下降一步」就直接更新 logits（见 9.3）。
- `g.label(name, values, options?)` 创建整数标签，shape 从 `values` 推断：`0` 是 `()`，`[0]` 是 `(1,)`，`[0, 1, 2]` 是 `(3,)`，`[[0], [1], [2]]` 是 `(3, 1)`。`tokens` 是每个类别的显示名，`classes` 是类别数，默认等于 `tokens.length`。没有 `tokens` 时必须给 `classes`，否则 tokens 面板和 3D 场景的目标列无法工作。

label 的 shape 要与使用它的运算一致：

| 运算 | 输入 shape | label 写法 | label shape |
|---|---|---|---|
| `nll_loss` 单样本 | `(C,)` | `g.label('t', 0, ...)` | `()` |
| `nll_loss` 批量 | `(N, C)` | `g.label('t', [0, 1, 2], ...)` | `(N,)` |
| `gather` 单样本 | `(C,)` | `g.label('t', [0], ...)` | `(1,)` |
| `gather` 批量 | `(N, C)` | `g.label('t', [[0], [1], [2]], ...)` | `(N, 1)` |

### 2.3 运算

运算是计算图上的方法，名字和参数顺序与 PyTorch 相同。torch 里按位置传的参数在这里按同样的顺序传；torch 里按关键字传的参数，以及节点的选项，放进最后一个对象：

```js
const z    = g.matmul(h, W.T, { name: 'z' });
const L    = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
const logp = g.sub(z, L, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
const loss = g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });
```

第一版内置运算：

| 运算 | 调用 | 说明 |
|---|---|---|
| `add` `sub` `mul` `div` | `g.add(a, b)` | 按 torch 规则广播 |
| `neg` `exp` `log` | `g.exp(a)` | 逐元素 |
| `matmul` | `g.matmul(a, b)` | 2 维乘 2 维，以及 1 维与 2 维的组合 |
| `transpose` | `g.transpose(a, dim0, dim1)` | 二维节点也可以写 `W.T` |
| `sum` `mean` | `g.sum(a)`、`g.sum(a, -1, { keepdim: true })` | 省略 `dim` 时对全部元素归约；`dim` 可以是数或数组 |
| `max` | `g.max(a, dim, { keepdim })` | 只返回最大值 |
| `logsumexp` | `g.logsumexp(a, dim, { keepdim })` | |
| `softmax` `log_softmax` | `g.softmax(a, dim)` | |
| `gather` | `g.gather(a, dim, index)` | `index` 是 label 节点，维数与 `a` 相同 |
| `nll_loss` | `g.nll_loss(input, target, { reduction, ignore_index })` | `reduction` 默认 `'mean'`，可选 `'sum'`、`'none'`；`ignore_index` 默认 `-100` |

选项对象里可以写的键：该运算的 torch 关键字参数（例如 `keepdim`、`reduction`），以及 `name`、`tex`、`elem`、`card`。写了其他键会抛错 `<运算名>: unknown option '<键>'`。`dim` 可以用负数。shape 不匹配在建图时就抛错，例如 `matmul: cannot multiply (1, 4) by (5, 4)`。

`W.T` 是二维节点的转置。第一次访问时创建名为 `W.T` 的 transpose 节点，之后每次访问返回同一个节点，所以 `g.matmul(h, W.T)` 和 `board(h, W.T, z)` 里的 `W.T` 是同一个节点。

### 2.4 节点命名规则

- 节点名在一张图里必须唯一，重复时抛错 `node '<名字>' already exists`。比较名字时忽略空白，`'log p'` 与 `'logp'` 算同一个名字。
- 运算不给 `name` 时自动命名为运算名加序号，例如 `exp1`、`sub2`。正文模板和刻度设置都要按名字引用节点，所以要在讲解里出现的运算都应当给 `name`。
- 名字会出现在步骤按钮、面板标题和卡片标题上，用读者能看懂的写法，例如 `'log p'`、`'LSE'`、`'W.T'`。
- 名字 `b` 有特殊含义：正文模板里的裸 `b` 表示当前样本下标。图里有名为 `b` 的节点（例如偏置）时，只要某段模板用到 `b`，创建 steps 或挂载面板时就抛错。偏置可以命名为 `bias`。
- 模板按最长前缀匹配名字，名字里可以有点和空格，例如 `W.T`、`log p`。

### 2.5 tex 与 elem

每个节点有两个 LaTeX 写法：

- `tex`：张量级写法，用在卡片公式和自动生成的反向正文里，例如 `\mathbf{h}`、`W^{\top}`、`\log\mathbf{p}`、`\ell`。
- `elem`：元素级的底，元素写成 `${elem}_{i}` 或 `${elem}_{k,i}`，下标从 1 开始，例如 `h_{2}`、`W^{\top}_{2,3}`、`\log p_{1}`。

默认值：名字是单个字母时，`elem` 是这个字母，`tex` 对非标量是 `\mathbf{字母}`，对标量是字母本身；名字更长时两者都是 `\mathrm{名字}`。名字含空格或不是单个字母时（例如 `'log p'`、`'loss'`），手动给出 `tex` 和 `elem`，否则卡片上显示成 `\mathrm{log p}` 这样的写法。JavaScript 字符串里的反斜杠要写两个：`'\\log\\mathbf{p}'`。

### 2.6 损失节点

反向传播从损失节点开始。没有用 `g.loss(node)` 指定时，库取最后创建的、元素个数为 1 的运算节点。如果在损失之后又创建了标量运算（例如为了显示而求和），损失节点会变成那个运算，这时用 `g.loss(loss)` 明确指定。图里没有元素个数为 1 的运算节点时，没有损失节点：没有反向步骤，梯度全为零，也不能挂 loss 面板。

### 2.7 读写数值

```js
loss.value.data[0];        // 当前损失
z.value.get([0, 2]);       // 按下标读一个元素
h.grad;                    // ∂ℓ/∂h，Tensor；label 的 grad 为 null
h.set([0.1, 0.2, 0.3, 0.4]);  // 只能对 input、param、label 调用，之后立即重算并通知
t.index();                 // label 的整数值
g.batch(() => { h.set(...); t.set(1); });  // 多次 set 只重算一次
g.reset();                 // 所有叶子恢复创建时的值
g.subscribe(() => { ... }); // 每次重算之后调用，返回取消订阅的函数
g.node('log p');           // 按名字找节点
g.jacobian(p, x);          // 当前值下 p 对 x 的 Jacobian，Tensor，不是节点
```

`Tensor` 的 `data` 是行主序的 `Float64Array`，`shape` 是数组。`g.jacobian(out, input, { row })` 给出 `row` 时只取第一维第 `row` 行的块，批量页用它显示当前样本的 Jacobian。

## 3. 讲解步骤

```js
import { createSteps } from '@yhnocoder/vizflow/steps';

const steps = createSteps(g, {
  overview: { title: '整张图', text: '...' },
  forward: [
    { show: [h, W.T], label: 'h, Wᵀ', title: '输入：h 和 Wᵀ', text: '...' },
    { show: z, active: [h, W.T, z], label: 'z = hWᵀ', title: 'lm_head：z = hWᵀ', text: '...' },
    { show: L, label: 'LSE', title: 'log-sum-exp', text: '...' },
    { show: logp, label: 'log p', title: 'log_softmax', text: '...' },
    { show: loss, label: 'ℓ', title: 'nll_loss', text: '...' },
  ],
  backward: 'auto',
});
```

### 3.1 前向步骤

- `overview` 是第 0 步，显示整张图。省略时标题为「总览」、正文为空。`label` 是步骤条上的按钮文字，默认等于 `title`。
- `forward` 按顺序列出前向各步。`show` 是这一步新出现的节点（一个或数组）；`active` 是这一步高亮的节点，默认等于 `show`；`title` 是正文标题；`text` 是正文，省略时正文为空；`label` 是按钮文字，默认等于 `title`。
- 第 k 步时，第 1 到 k 步 `show` 过的节点都出现，`active` 的节点高亮。
- 所有 `show` 的并集是 `steps.visible`。没有放进任何一步 `show` 的中间节点在 3D 场景里不画，也不成为反向步骤。标签节点（label）不需要放进 `show`。

### 3.2 反向步骤

`backward` 省略或写 `'auto'` 时，反向步骤由库生成：

1. `visible` 中求梯度的节点（label 除外，损失之后创建的节点除外）按创建顺序倒序排列。
2. 损失节点和它前面第一个可见节点合成第一步，之后每个可见节点一步。
3. 默认标题是 `∂ℓ/∂节点名`（合成的步骤用其中的非损失节点），按钮文字是 `∂节点名`，正文是该节点反向层名卡片的链式法则公式。

lm_head 页因此得到五步：`ℓ` 与 `log p`、`LSE`、`z`、`W.T`、`h`。再举一个小例子：

```js
const x = g.input('x', ['V'], [2.0, 1.0, 0.2, -0.6, -1.4]);
const t = g.label('t', 0, { classes: 5 });
const logp = g.log_softmax(x, -1, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
const loss = g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });
```

`log_softmax` 在图上是一个节点，不拆成更小的运算。`x`、`logp`、`loss` 都在 `show` 里时，反向有两步：第 1 步是 `loss` 与 `log p`，标题 `∂ℓ/∂log p`，按钮 `∂log p`；第 2 步是 `x`，标题 `∂ℓ/∂x`，按钮 `∂x`。

要调整时写成对象：

```js
backward: {
  auto: true,
  merge: [[logy, y]],
  override: {
    1: { label: '∂y', title: '从损失出发', text: '...' },
    3: { title: '传到分母 S' },
  },
},
```

- `merge`：每组节点并入组内最早出现的那一步。上面把 `log y` 和 `y` 放进同一步。
- `override`：按反向步骤的序号（从 1 开始）覆盖 `title`、`text`、`label`，没写的字段用自动内容。序号超出步骤数时抛错。

先用 `'auto'` 运行一次，看步骤条上反向有几步、每步是哪个节点，再写 `override` 的序号。图里没有损失节点时，反向是 0 步；这时写了 `merge` 或 `override` 会抛错。

### 3.3 当前步骤与批量样本

```js
steps.go(3);         // 跳到 steps.list 的第 3 项（总览是 0）
steps.next(); steps.prev();
steps.current;       // 当前项在 steps.list 里的位置
steps.setFocus(1);   // 批量模式下选中第 2 个样本
steps.focus;         // 当前样本下标，默认 0
steps.text();        // { title, body }，模板已求值，公式未渲染
```

`steps.list` 依次是总览、前向各步、反向各步，步骤条上的按钮也按这个顺序排列，最后一个按钮是反向的最后一步。`setFocus(b)` 的 `b` 必须是 0 到「visible 中节点第一维的最大值减 1」之间的整数，否则抛错。切换样本只改变 3D 场景里高亮的行和模板里 `b` 的值，不改变计算图的数值。批量页通常自己画一组样本按钮调用 `steps.setFocus`，并用 `steps.subscribe` 更新按钮状态（见 `examples/softmax-ce.html` 的 `drawSamples`）。在 3D 场景里点击批量张量某一行的柱子也会切换样本。

## 4. 正文模板

步骤的 `title` 与 `text`、损失面板的 `text`、各面板的 `note` 都是 HTML 字符串，其中：

- `$...$` 是行内公式，`$$...$$` 是独立成行的公式，由 KaTeX 渲染。要显示美元符号本身时写 `\\$`。
- `{{...}}` 是数值模板，显示计算图上的当前值，输入变化后自动更新。

### 4.1 语法

```
模板   := '{{' 表达式 (':' 格式)? '}}'
表达式 := 引用 (('+' | '-') 整数)?
引用   := 'b' | 名字 ('.grad')? ('[' 下标 (',' 下标)* ']')?
下标   := 整数 | 'b' | 名字 | 名字 '[' 'b' ']'
格式   := 整数（小数位数） | 'token'
```

| 写法 | 含义 |
|---|---|
| `{{loss}}` | 标量节点的值，可以省略下标 |
| `{{loss:3}}` | 保留 3 位小数 |
| `{{z[2]}}` | `z` 的第 3 个元素，下标从 0 开始 |
| `{{z[t]}}` | 用 label `t` 的整数值作下标，对一维、二维节点都适用；label 自身有多个元素时取第一个元素，批量时写 `t[b]` |
| `{{t}}`、`{{t+1}}` | label 的整数值；`+1` 用于显示从 1 开始的序号 |
| `{{t:token}}` | `tokens[t]` |
| `{{z.grad[t]}}` | 梯度 |
| `{{W.T.grad[0, t]}}` | 节点 `W.T` 的梯度，二维下标 |
| `{{log p[t]}}` | 名字里有空格，按名字匹配 |
| `{{z.grad[t]+1:3}}` | 值加 1 后保留 3 位小数 |
| `{{b+1}}` | 当前样本序号（批量页） |
| `{{X[b, t[b]]}}` | 第 b 行、第 `t[b]` 列；`t[b]` 取 label 第 b 行的值 |

规则：

- 模板内的空白先全部去掉，再按图中节点名做最长前缀匹配。所以 `{{log p[t]}}` 匹配节点 `log p`，`{{W.T.grad[0,t]}}` 匹配节点 `W.T` 再加 `.grad`。
- 非标量节点必须写下标，下标个数等于维数。例外：shape 为 `(1, n)` 的节点只写一个下标时视为 `[0, i]`，`(n, 1)` 视为 `[i, 0]`。
- 偏移只能是整数，加在数值上，例如 `{{t+1}}`。模板里不能写任意表达式，不执行 JavaScript。
- 不写格式时，数值按统一规则显示：绝对值不小于 100 保留 0 位，不小于 10 保留 1 位，不小于 1 保留 2 位，其余保留 3 位。label 的值按整数显示。
- 在 `$...$` 里的数用 ASCII 负号 `-`，公式外用 `−`（U+2212）。
- `b` 是当前样本下标 `steps.focus`，按整数输出，可以加偏移和小数位格式，不能写 `:token` 或下标。没有挂载 steps 的面板里 `b` 为 0。

### 4.2 与 LaTeX 花括号相邻

LaTeX 的花括号是单个的，模板是两个，`\mathbf{h}`、`W^{\top}_{k,t}` 不会被当成模板。模板紧贴 LaTeX 花括号时会出现三个花括号，库无法判断边界，直接报错。在中间留一个空格：

```js
'$$z_{ {{t+1}} } = {{z[t]}}$$'          // 正确
'$$z_{{{t+1}}} = {{z[t]}}$$'            // 报错：starts with '{{{'
'$$\\frac{ {{u[t]}} }{ {{S}} }$$'      // 正确
```

### 4.3 模板的范围与报错时机

模板只能读取节点的值与梯度。模板在 `createSteps` 和 `mountPanels` 时编译，名字不存在、下标个数不对、对 label 取 `.grad`、对没有 tokens 的节点取 `:token` 都在这时抛出带模板原文的错误，例如：

```
cannot parse template '{{x}}': node 'x' with shape (5) needs a subscript
cannot parse template '{{q[0]}}': no node name matches 'q[0]'
```

下标越界在求值时才发现（例如 label 改变后），同样抛出带模板原文的错误。

## 5. 面板与步骤条（ui）

```js
import { mountSteps, mountPanels } from '@yhnocoder/vizflow/ui';

mountSteps('#steps', steps);
mountPanels('#side', g, [ ...面板 ], { steps });
```

### 5.1 mountSteps

`mountSteps(el, steps)` 在容器里画出步骤条（前向、反向两组按钮，左右箭头）和当前步骤的标题与正文。键盘左右方向键切换步骤；焦点在输入框、滑块或步骤条以外的按钮上时不响应。`el` 可以是选择器字符串或元素，找不到时抛错。

### 5.2 mountPanels

`mountPanels(el, g, items, { steps })` 按顺序画出一列面板卡片。传入 `steps` 时，面板跟随当前样本，模板里的 `b` 有值，matrix 函数能拿到当前步骤状态。返回值和其他 mount 一样有 `dispose()`。

**loss**：损失读数，下方可选一行说明和一组按钮。点「更新一步」后画出损失曲线，其他任何输入变化都会清空曲线。

```js
{ loss: {
    sgd: { params: [W], lr: 0.5, label: '更新一步 W' },   // θ ← θ − lr·∇θ ℓ；按钮文字后自动加「（η = 0.5）」
    random: { nodes: [h], range: [-2, 2], label: '随机' }, // 在 range 内随机取值（两位小数）
    reset: true,                                         // 「重置」按钮，调用 g.reset()
    text: '目标 "{{t:token}}"，log p = {{log p[t]:3}}',  // 模板，可以含公式
} }
```

`sgd.params` 与 `random.nodes` 可以是 input 或 param 节点。所有字段都可以省略；只要读数时写 `'loss'`。读数保留 3 位小数。图里没有损失节点时挂载 loss 面板会抛错 `graph has no loss node`。`sgd.label` 省略时为「更新一步 节点名」，`random.label` 省略时为「随机 节点名」。

**sliders**：一个节点的每个元素一个滑块。

```js
{ sliders: h, min: -2.5, max: 2.5, step: 0.01, title: '隐状态 h，(1, d)' }
```

`min`、`max`、`step` 默认 −2.5、2.5、0.01。标题默认为「节点名，shape」。每个滑块的无障碍名称是节点名加下标数字，例如 `h₁`、`x₃`。二维节点只显示当前样本那一行。

**tokens**：类别表，每行一个类别，单选按钮设为目标类。

```js
{ tokens: t, columns: [z, logp, { node: z, grad: true, title: '∂ℓ/∂z' }], title: '词表与目标 token' }
{ tokens: t, sliders: x, min: -6, max: 6, columns: [p, { node: x, grad: true, title: 'p − y' }] }
```

`columns` 中每一项是节点（显示值）或 `{ node, grad, title }`（`grad: true` 显示梯度，按正负着色）；数值列保留 3 位小数。给了 `sliders` 时每行带一个滑块，用来按类别调 logits，`min`、`max` 默认 −2.5、2.5。列的长度要等于类别数。token 的前导空格显示为 `␣`。label 既没有 tokens 也没有 classes 时抛错。

**matrix**：矩阵热力图，单元格保留 2 位小数，按正负着色。

```js
{ matrix: W.T, highlight: t, title: 'Wᵀ' }
{ matrix: W.T, grad: true, highlight: t, title: '∂ℓ/∂Wᵀ', note: '...' }
{ matrix: (graph, state) => graph.jacobian(p, x, { row: state.focus }),
  highlight: { node: t, axis: 'row' }, title: 'Softmax 的 Jacobian', note: '...' }
```

`matrix` 是节点或函数。函数接收计算图和当前步骤状态（`state.focus` 是当前样本），返回 `Tensor`，用于显示不在计算图里的矩阵，例如 Jacobian。结果必须是二维的，否则抛错 `matrix panel: ... must be 2-D`。`highlight` 是 label 时框出目标列，写成 `{ node, axis: 'row' }` 时框出目标行。label 有 tokens 且类别数等于列数时，列表头显示 token，否则显示从 1 开始的序号。

**code**：代码卡片，`#` 到行尾显示为注释色。

```js
{ code: 'p = F.softmax(x, dim=-1)\nloss = F.nll_loss(p.log(), t)', title: '对应的 PyTorch', note: '...' }
```

**note**：一段说明文字，可以写公式和模板。

```js
{ note: '当前目标类是 {{t+1}}，$p_t = {{p[t]:3}}$', title: '说明' }
```

面板放在哪里由页面决定。同一张图可以挂多次 `mountPanels`，分别放进不同容器，例如 softmax-ce 页把 loss 面板和其他面板放在两个容器里。

### 5.3 页面里自己的文字

标题下的导语等页面自己的文字如果含公式，用 formula 入口的 `renderText` 渲染：

```js
import { renderText } from '@yhnocoder/vizflow/formula';
renderText('从隐状态 $\\mathbf{h}$ 经过 lm_head 得到 logits。', document.getElementById('lede'));
```

`renderText` 不处理 `{{...}}` 模板。需要随数值更新的文字用 note 面板。

## 6. 3D 场景（scene3d）

```js
import { mountScene, board, stack, node } from '@yhnocoder/vizflow/scene3d';

const scene = mountScene('#stage', g, {
  steps,
  layout: [
    board(h, W.T, z),
    stack(z, L, logp, loss),
  ],
  hint: '点 logit 柱子换目标 token · 悬停层名看 shape · 反向时悬停连线',
  camera: { direction: [0.62, 0.5, 1], azimuth: [-0.75, 0.6], polar: [0.55, 1.5] },
});
scene.resetView();
scene.dispose();
```

### 6.1 选项

- `steps`：给出时场景按当前步骤显示和高亮节点，只画 `steps.visible` 中的节点。省略时全部显示、全部高亮。
- `layout`：布局模板列表，见 6.2。
- `hint`：图例下方的一行灰色提示，省略时不显示。画布宽度不超过 600px 时隐藏。
- `camera`：相机。`direction` 是从观察点指向相机的方向，默认 `[0.42, 0.78, 1]`；`azimuth` 是相对初始方向的水平旋转范围（弧度），默认 `[-0.8, 0.7]`；`polar` 是极角范围，默认 `[0.3, 1.45]`；`zoom` 是缩放范围，默认 `[0.6, 3]`。取景按所有元素和标签的范围自动计算，内容尽量填满画布，标签离画布边缘至少 12px，不需要调。
- `labels`：按节点名给标签加像素偏移 `[dx, dy]`（x 向右、y 向下），作用于柱子块的块名标签和标量节点的标签，例如 `labels: { 'log p': [12, -8] }`。给了偏移的标签固定在这个位置，不再翻转和避让，其他标签避开它。一般由编辑模式写出，见 6.5。
- `edit`：编辑模式，见 6.5。只在写页面时打开，发布前删掉。

场景左上角固定显示图例，右上角是「复位视角」按钮。浏览器创建不了 WebGL 时，容器里显示一行提示，`mountScene` 照常返回，步骤条和面板不受影响。

### 6.2 布局模板

一个节点只放置一次。已经被前面的模板放置的节点出现在后面的 `stack` 里时作为起点，不重复放置。

**board(a, b, out, options?)**：矩阵乘专用，按课本摆法把 `a`、`b`、`out = a @ b` 排在同一块地面上：`b` 在后方，`a` 在左前方，`out` 在 `b` 的正前方，与 `b` 的列对齐。`b` 必须是二维的。

**stack(...nodes, options?)**：把节点按计算顺序往上叠。

- 每行元素数大于 1 的节点画成一层柱子，放在上一层上方，列与上一层对齐，并比上一层往后移 `back`，斜视时各层的柱子和标签错开；批量时第 1 行在最前面，后面的行依次往后排开。
- 每行只有一个元素的节点画成小圆柱：由归约或 softmax 类运算得到的（例如 LSE、S）放在两层之间的侧面；由 `gather`、`nll_loss` 选出的（例如 `y`、损失）放在目标列正上方，随目标类移动；由它们经逐元素运算得到的节点（例如批量时每个样本的 `log y`、`l_b`）每行跟随自己那一行的目标列，对批量取平均后的损失放在中间；损失节点画成扁方块。
- 节点和选项对象可以混写，选项对象可以放在任意位置。

**node(n, { at })**：把一个节点单独放在 `at` 位置，用于每行只有一个元素的节点。每行多于一个元素的节点放进 `node` 会抛错，提示改用 `stack`；多列节点用 `stack` 或 `board`。

模板选项：

| 选项 | 适用 | 含义 | 默认 |
|---|---|---|---|
| `at` | 全部 | 起点坐标 `[x, y, z]` | `[0, 0, 0]` |
| `gap` | board | 行距 | 0.7 |
| `gap` | stack | 列距（没有可对齐的上一层时） | 1.45 |
| `level` | stack | 相邻两层的高度差 | 5.2 |
| `back` | stack | 每一层比上一层往后（$z$ 减小）移动的距离；侧面的标量节点往后移一半 | 0.7 |
| `scale` | board、stack | 同时缩放间距，柱子底面尺寸不变 | 1 |
| `scales` | board、stack | 按节点名指定柱子刻度 | 由初始值决定 |
| `offsets` | stack | 按节点名把这个 stack 放置的一层整体平移 `[dx, dy, dz]`，后面以它为起点的层跟着移动 | 无 |

### 6.3 柱子刻度

每个节点前向、反向各有一个固定刻度，挂载时由初始值决定：初始值中的最大绝对值对应 1.6 个单位高。之后拖动滑块时刻度不变，柱子高度超过 2.0 个单位时截断。初始值很小、之后会变大的节点（例如梯度），或者想让几个节点高度可比时，用 `scales` 直接指定，单位是「每单位数值对应的高度」：

```js
stack(x, u, S, p, y, logy, loss, {
  scales: { x: { forward: 0.3, backward: 1.6 }, u: { forward: 0.11, backward: 4 }, 'log p': { forward: 0.32 } },
})
```

`scales` 的键是节点名原文，包括空格和点，例如 `'log p'`、`'W.T'`。

### 6.4 交互与卡片

- 悬停柱子显示元素名和当前值。可以点击设目标的柱子，提示末尾写「点击设为目标 token」（label 给了 `tokens`）或「点击设为目标类」（只给了 `classes`）。
- 数值标签只在当前步骤的节点上显示，同一个节点的数值保留相同的小数位数；批量时数值标签和标量节点的标签只显示选中的样本。
- 标签互不重叠，并且离画布边缘至少 12px：层名和节点标签与其他标签相交时会挪开，节点标签超出画布右边缘时翻到节点左侧；其他标签相交或超出时隐藏。
- 悬停层名显示层名卡片：前向是公式、shape、真实模型尺寸、当前值；反向是链式法则展开、Jacobian 的 shape、当前梯度。
- 反向步骤时悬停连线显示连线卡片：局部导数、上游梯度、两者乘积；节点有多条出边时说明各条相加。上游梯度恒为零的连线画成虚线。
- 点击类别对齐的张量（列数等于类别数、经 `nll_loss` 或 `gather` 与 label 相连的张量及其上游同列张量）的柱子，把该列设为目标类。批量时点击某一行的柱子同时切换样本。

卡片内容由库生成。要改写某个节点的层名卡片，在建图时给出 `card`，没写的字段用自动内容：

```js
const x = g.input('x', ['V'], [2.0, 1.0, 0.2, -0.6, -1.4], {
  card: { forward: { title: 'logits', note: '模型最后一层的输出，每个类别一个实数。' } },
});
const S = g.sum(u, -1, { keepdim: true, name: 'S', tex: 'S',
  card: { forward: { title: '分母', tex: ['S = \\sum_j u_j'] }, backward: { note: '...' } } });
```

`card.forward` 与 `card.backward` 各有 `title`（文字）、`tex`（LaTeX 数组，每项一行）、`note`（文字）。

### 6.5 编辑模式

自动布局不满意时，临时在 `mountScene` 的选项里加上 `edit: true`，用 `npm run dev` 打开页面，画布右下角出现「编辑布局」面板：

```js
const scene = mountScene('#stage', g, {
  steps,
  edit: true,
  layout: [
    board(h, W.T, z),
    stack(z, L, logp, loss),
  ],
});
```

- 拖动块：按住柱子、圆柱或底板在水平面内拖动。`board` 的三个块一起移动，写成 `board` 的 `at`；`node` 写成它的 `at`；`stack` 的第一层（没有从前面模板接过来的起点时）写成 `stack` 的 `at`，其余层写成 `stack` 的 `offsets`。
- 调间距：面板里改 `board` 的 `gap`、`scale` 和 `stack` 的 `level`、`back`，场景立即重排。
- 移动标签：拖动块名或标量节点的标签，写成 `labels`。
- 「设为默认视角」：先用鼠标转到想要的角度，再点这个按钮，写成 `camera.direction`。
- 「复制配置」：面板里出现一段代码并复制到剪贴板，例如：

```js
layout: [
  board(g.node('h'), g.node('W.T'), g.node('z')),
  stack(g.node('z'), g.node('LSE'), g.node('log p'), g.node('loss'), { offsets: { 'log p': [1.2, 0, -0.6] } }),
],
camera: { direction: [0.3, 0.8, 1] },
labels: { z: [24, -10] },
```

用它替换页面里 `mountScene` 选项中的 `layout`（以及 `camera`、`labels`），节点写成 `g.node('节点名')`，图的变量名不是 `g` 时改成实际的名字，也可以换回自己的变量名。然后删掉 `edit: true`。编辑模式不保存任何东西，刷新页面后回到页面代码里的配置，所以改完要先复制。每次拖完或改完数值，视角会回到默认方向并重新取景，看到的就是粘贴回去以后的样子。

## 7. 主题与页面布局

ui 和 scene3d 自动注入样式与字体。只用 core、steps 的页面需要样式时写 `import '@yhnocoder/vizflow/theme';`。

- 颜色变量：`--bg` `--surface` `--ink` `--fg` `--muted` `--line` `--line-strong` `--pos`（正值，橙）`--neg`（负值，蓝），字体变量 `--serif` `--sans` `--mono`。页面自己的样式也用这些变量，深色模式会跟着切换。
- 深浅色默认跟随系统。`<html data-theme="light">` 或 `data-theme="dark"` 强制使用其中一套，运行时修改这个属性时 3D 场景会重新读取颜色。
- 布局类：`.vf-page`（页面容器，最大宽 1360px）；`.vf-split`（左侧自适应，右侧面板 360px，980px 以下改为单列）；`.vf-stack`（单列）；`.vf-inline`（与正文同宽，用于文章里的小部件）。
- 尺寸变量：`--vf-side-width`（右侧宽度，默认 360px）、`--vf-gap`（间距，默认 20px）、`--vf-stage-height`（画布高度，默认 `min(68vh, 640px)`，980px 以下 62vh）。
- `readTheme()` 返回线性空间的颜色，`onThemeChange(fn)` 在主题变化时调用，两者在 theme 入口，页面自己画图时使用。

## 8. 自定义运算

内置运算不够时用 `defineOp` 注册新运算，注册后它出现在所有计算图上。下面的例子注册逐元素平方：

```js
import { defineOp, Tensor } from '@yhnocoder/vizflow/core';
import { registerExplain } from '@yhnocoder/vizflow/explain';

defineOp({
  name: 'square',
  args: ['input'],          // torch 的位置参数名，按顺序
  tensors: ['input'],       // 其中哪些是张量
  shape: ([a]) => [...a],
  forward: ([a]) => new Tensor(a.shape, a.data.map((v) => v * v)),
  backward: (gradOut, [a]) => [new Tensor(a.shape, gradOut.data.map((g, i) => 2 * a.data[i] * g))],
});

registerExplain('square', {
  pattern: 'elementwise',
  local: (ctx, _out, _arg, input) => 2 * ctx.inputs[0].get(input),
  tex: { forward: (s) => `${s.in[0]}^{2}`, local: (s) => `2${s.in}` },
});

const g = graph({ check: true });
const x = g.input('x', [3], [1, 2, 3]);
const y = g.square(x, { name: 'y' });
```

- `defaults` 写 torch 里按关键字传的参数及默认值，它们出现在选项对象里；`differentiable` 标出每个张量参数是否求梯度（例如整数下标为 `false`）。
- 讲解部分（`explain`）可以写在 `defineOp` 的 `explain` 字段里，也可以用 `registerExplain` 单独注册。`pattern` 从 `'elementwise'`、`'reduce'`、`'dense'`、`'contract'`、`'select'` 中选一个，决定连线怎么画。`local` 返回一条边的局部导数，`tex.forward` 与 `tex.local` 给出卡片上的公式。接口定义见设计文档「定义新运算」与「连线模式」两节。
- 只有数值、没有讲解的运算可以用在只引入 core 的场合。用到 steps 或 scene3d 时必须有讲解，否则抛错 `explain: op '<名字>' of node '<节点>' has no explain definition`。
- 同名运算只能注册一次。运算名不能与计算图已有的成员同名（例如 `input`、`loss`、`node`）。
- 写自定义运算时打开 `graph({ check: true })`，`backward` 写错会立即抛出 `gradient check failed`。

TypeScript 页面用声明合并补充类型：

```ts
import type { Node, OpOptions } from '@yhnocoder/vizflow/core';

declare module '@yhnocoder/vizflow/core' {
  interface Graph {
    square(input: Node, options?: OpOptions): Node;
  }
}
```

## 9. 三个完整示例

### 9.1 lm_head 与 NLL Loss（`examples/lm-head-nll.html`）

HTML 只有 `#stage`（画布）、`#steps`（步骤条）、`#side`（面板）三个容器，放在 `.vf-split` 里。

**建图**：隐状态 `h`（1×4）乘 lm_head 权重的转置得到 logits `z`（1×5），按 PyTorch 的做法用 logsumexp 与减法写出 log_softmax，再接 nll_loss。`real` 让层名卡片显示 Qwen3-8B 的真实尺寸。

```js
const g = graph({
  dims: { d: 4, V: 5 },
  real: { d: 4096, V: 151936, name: 'Qwen3-8B' },
});
const h = g.input('h', [1, 'd'], [0.8, -1.1, 0.5, 1.4]);
const W = g.param('W', ['V', 'd'], [
  [ 0.9, -0.6,  0.3,  0.6],
  [ 0.2, -0.8,  0.6,  0.2],
  [-0.4,  0.2,  0.9,  0.3],
  [ 0.5,  0.7, -0.6, -0.2],
  [-0.7,  0.4, -0.3, -0.5],
], { tex: 'W', elem: 'W' });
const t = g.label('t', [0], { tokens: [' mat', ' floor', ' sofa', ' dog', ' sat'] });

const z    = g.matmul(h, W.T, { name: 'z' });
const L    = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
const logp = g.sub(z, L, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
const loss = g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });
```

这里 `logp` 的 shape 是 `(1, 5)`，所以 label 写成 `[0]`，shape `(1,)`，与 `nll_loss` 的 `(N, C)` 配 `(N,)` 对应。

**步骤**：总览加五个前向步骤；反向自动生成五步，用 `override` 改写每一步的标题和正文。正文里用模板代入目标列的数值。

```js
const steps = createSteps(g, {
  overview: {
    title: '整张图',
    text: '<p>...目标 token 是 <span class="ctx">{{t:token}}</span>。</p>'
      + '$$\\ell = -\\log\\operatorname{softmax}(\\mathbf{h}W^{\\top})_t = \\mathrm{LSE}(\\mathbf{z}) - z_t = {{loss:3}}$$',
  },
  forward: [
    { show: [h, W.T], label: 'h, Wᵀ', title: '输入：h 和 Wᵀ',
      text: '<p>...</p>'
        + '$$\\mathbf{h} = ({{h[0]:2}},\\; {{h[1]:2}},\\; {{h[2]:2}},\\; {{h[3]:2}}) \\in \\mathbb{R}^{1\\times 4}$$' },
    { show: z, active: [h, W.T, z], label: 'z = hWᵀ', title: 'lm_head：z = hWᵀ',
      text: '<p>...</p>'
        + '$$z_{ {{t+1}} } = \\sum_k h_k W^{\\top}_{k, {{t+1}} } = ({{h[0]:2}})({{W[t, 0]:2}}) + \\dots = {{z[t]}}$$' },
    { show: L, label: 'LSE', title: 'log-sum-exp',
      text: '<p>...</p>$$\\mathrm{LSE} = m + \\log\\sum_j e^{z_j - m} = {{LSE}}$$' },
    { show: logp, label: 'log p', title: 'log_softmax',
      text: '<p>...</p>$$\\log p_{ {{t+1}} } = {{z[t]}} - {{LSE}} = {{log p[t]}}$$' },
    { show: loss, label: 'ℓ', title: 'nll_loss',
      text: '<p>...</p>$$\\ell = -\\log p_t = {{loss:3}}$$' },
  ],
  backward: {
    auto: true,
    override: {
      1: { title: '从 nll 出发', text: '...' },
      2: { title: '传到 LSE', text: '...$$\\frac{\\partial\\ell}{\\partial\\mathrm{LSE}} = {{LSE.grad}}$$' },
      3: { title: '传到 logits', text: '...$$g_{ {{t+1}} } = {{z.grad[t]+1:3}} - 1 = {{z.grad[t]:3}}$$' },
      4: { title: '传到 lm_head 权重', label: '∂Wᵀ', text: '...' },
      5: { title: '传到隐状态 h', text: '...$$\\frac{\\partial\\ell}{\\partial\\mathbf{h}} = ({{h.grad[0]:3}},\\; \\dots)$$' },
    },
  },
});
```

`{{W[t, 0]}}` 读的是 `W`（V×d）的第 t 行，对应 `W.T` 的第 t 列。`{{z.grad[t]+1:3}}` 是梯度加 1，即 $p_t$。

**布局**：`board` 把 `h`、`W.T`、`z` 摆成矩阵乘，`stack` 从 `z` 往上叠 LSE、log p 和损失。`z` 已被 board 放置，在 stack 里作为起点。`scales` 让前向和反向的柱子高度合适。

```js
mountSteps('#steps', steps);
const scene = mountScene('#stage', g, {
  steps,
  layout: [
    board(h, W.T, z, { scales: { h: { forward: 0.45, backward: 1.2 }, 'W.T': { forward: 0.6, backward: 0.8 }, z: { forward: 0.3, backward: 1.6 } } }),
    stack(z, L, logp, loss, { scales: { 'log p': { forward: 0.32, backward: 1.6 } } }),
  ],
  hint: '点 logit 柱子换目标 token · 悬停层名看 shape · 反向时悬停连线',
});
```

**面板**：损失读数带「更新一步 W」「随机 h」「重置」三个按钮；`h` 的滑块；类别表并排显示 `z`、`log p` 和 `∂ℓ/∂z`；`W.T` 的值和梯度两个热力图，框出目标列；对应的 PyTorch 代码。

```js
mountPanels('#side', g, [
  { loss: {
      sgd: { params: [W], lr: 0.5, label: '更新一步 W' },
      random: { nodes: [h], range: [-2, 2] },
      reset: true,
      text: '目标 "{{t:token}}"，log p = {{log p[t]:3}}',
  } },
  { sliders: h, min: -2.5, max: 2.5, step: 0.01, title: '隐状态 h，(1, d)' },
  { tokens: t, columns: [z, logp, { node: z, grad: true, title: '∂ℓ/∂z' }], title: '词表与目标 token' },
  { matrix: W.T, highlight: t, title: 'Wᵀ，(d, V) = (4, 5)' },
  { matrix: W.T, grad: true, highlight: t, title: '∂ℓ/∂Wᵀ = hᵀg，(4, 5)', note: '...' },
  { code: PYTORCH, title: '对应的 PyTorch', note: '...' },
], { steps });
```

页面末尾写 `document.body.dataset.ready = 'true'`，供 e2e 测试等待（见第 10 节）。

### 9.2 Softmax 与交叉熵（`examples/softmax-ce.html`）

这一页有单样本 `(V,)` 和批量 `(B, V)` 两种模式，各建一张图，用标签页切换。每种模式由一个函数返回图、步骤、布局与面板配置，切换时 `dispose()` 旧的挂载再挂新的。

**单样本建图**：softmax 拆成 exp、sum、div 三步，`gather` 取出目标类概率，再取负对数。`card` 改写层名卡片的标题和说明。

```js
const g = graph({ dims: { V } });
const x = g.input('x', ['V'], LOGITS[0], {
  card: { forward: { title: 'logits', note: '模型最后一层的输出，每个类别一个实数。' } },
});
const t = g.label('t', [TARGETS[0]], { classes: V });
const u = g.exp(x, { name: 'u', card: { forward: { title: '取指数', note: '...' } } });
const S = g.sum(u, -1, { keepdim: true, name: 'S', tex: 'S', card: { forward: { title: '分母', note: '...' } } });
const p = g.div(u, S, { name: 'p', card: { forward: { title: '概率', note: '...' } } });
const y = g.gather(p, -1, t, { name: 'y', tex: 'y', card: { forward: { title: '目标类概率', note: '...' } } });
const logy = g.log(y, { name: 'log y', tex: '\\log y', elem: '\\log y' });
const loss = g.neg(logy, { name: 'loss', tex: '\\ell', elem: '\\ell', card: { forward: { title: '交叉熵损失', note: '...' } } });
```

`gather` 要求 label 的维数与输入相同，所以单样本的 label 是 `[0]`（shape `(1,)`），批量的是 `[[0], [1], [2]]`（shape `(3, 1)`）。label 没有 tokens，用 `classes: V` 给出类别数。

**单样本步骤**：最后一个前向步骤同时显示 `y`、`log y`、`loss`；反向把 `log y` 并入 `y` 的那一步。

```js
const steps = createSteps(g, {
  overview: { label: '总览', title: '整张图', text: '...$$\\ell=-\\log\\frac{e^{x_t}}{\\sum_j e^{x_j}}={{loss:3}}$$' },
  forward: [
    { show: x, label: 'x', title: 'logits', text: '...$$\\mathbf{x}=({{x[0]:2}},\\;{{x[1]:2}},\\;\\dots)$$' },
    { show: u, label: 'u = eˣ', title: '取指数', text: '...$$u_{ {{t+1}} }=e^{ {{x[t]:2}} }={{u[t]}}$$' },
    { show: S, label: 'S', title: '求和', text: '...$$S={{S}}$$' },
    { show: p, label: 'p', title: '归一化', text: '...$$p_{ {{t+1}} }=\\frac{ {{u[t]}} }{ {{S}} }={{p[t]:3}}$$' },
    { show: [y, logy, loss], label: 'ℓ', title: '取目标类，取负对数', text: '...' },
  ],
  backward: {
    auto: true,
    merge: [[logy, y]],
    override: {
      1: { label: '∂y', title: '从损失出发', text: '...$$\\frac{\\partial\\ell}{\\partial y}=-\\frac{1}{ {{y:3}} }={{y.grad}}$$' },
      2: { title: '传到概率层', text: '...' },
      3: { title: '传到分母 S', text: '...' },
      4: { title: '传到 u', text: '...' },
      5: { title: '传到 logits', text: '...' },
    },
  },
});
```

**批量建图与步骤**：输入是 `(3, 5)`，`S` 是 `(3, 1)`，每个样本的损失 `l_b` 是 `(3, 1)`，最后 `mean` 得到标量损失。正文用 `b` 代入当前样本：

```js
const g = graph({ dims: { B, V } });
const X = g.input('X', ['B', 'V'], LOGITS, { card: { ... } });
const t = g.label('t', TARGETS.map((v) => [v]), { classes: V });
// U、S、P、y、log y 与单样本相同，只是名字改为大写
const lb = g.neg(logy, { name: 'l_b', tex: '\\ell_b', elem: '\\ell', ... });
const loss = g.mean(lb, { name: 'loss', tex: '\\ell', elem: '\\ell', ... });

const sample = '下面代入的是样本 {{b+1}}（目标类 {{t[b]+1}}），可在右侧切换样本。';
// 正文例：'$$P_{ {{b+1}},{{t[b]+1}} }=\\frac{ {{U[b,t[b]]}} }{ {{S[b]}} }={{P[b,t[b]]:3}}$$'
// 反向：merge: [[lb, logy, y]]
```

**布局**：一个 `stack` 从 logits 往上叠到损失，层高 3.9，每层往后移 1.7（批量时 2.2），让三层板在斜视角下沿对角线错开；`camera` 给出这一页自己的视角。

```js
const CAMERA = { direction: [0.62, 0.5, 1], azimuth: [-0.75, 0.6], polar: [0.55, 1.5] };
layout: [stack(x, u, S, p, y, logy, loss, {
  level: 3.9,
  back: 1.7,
  scales: { x: { forward: 0.3, backward: 1.6 }, u: { forward: 0.11, backward: 4 }, p: { forward: 2.4, backward: 0.5 } },
})],
```

**面板**：loss 面板单独挂在 `#side-loss`，其他面板挂在 `#side`。类别表每行带一个 logit 滑块；Jacobian 由函数计算，批量时用 `state.focus` 取当前样本那一块，框出目标行。

```js
loss: {
  sgd: { params: [x], lr: 1, label: '梯度下降一步' },
  random: { nodes: [x], range: [-3, 3], label: '随机' },
  reset: true,
  text: '目标类 {{t+1}}，$y=p_{ {{t+1}} }={{y:3}}$',
},
side: [
  { tokens: t, sliders: x, min: -6, max: 6, columns: [p, { node: x, grad: true, title: 'p − y' }], title: 'logits 与目标类' },
  { matrix: (graph) => graph.jacobian(p, x), highlight: { node: t, axis: 'row' }, title: 'Softmax 的 Jacobian', note: '...' },
  { code: PYTORCH_SINGLE, title: '对应的 PyTorch' },
],
```

**挂载与切换**：

```js
function show(name) {
  mounted.forEach((m) => m.dispose());
  const mode = modes[name];
  mounted = [
    mountScene('#stage', mode.g, { steps: mode.steps, layout: mode.layout, camera: CAMERA, hint: mode.hint }),
    mountSteps('#steps', mode.steps),
    mountPanels('#side-loss', mode.g, [{ loss: mode.loss }], { steps: mode.steps }),
    mountPanels('#side', mode.g, mode.side, { steps: mode.steps }),
  ];
}
```

### 9.3 只用面板的小部件（`examples/widget-softmax.html`）

不需要 3D 时只引入 core、steps、ui，页面不会加载 three.js：

```js
import { graph } from '@yhnocoder/vizflow/core';
import { createSteps } from '@yhnocoder/vizflow/steps';
import { mountPanels, mountSteps } from '@yhnocoder/vizflow/ui';

const g = graph({ dims: { V: 5 } });
const x = g.input('x', ['V'], [2.0, 1.0, 0.2, -0.6, -1.4]);
const t = g.label('t', 0, { tokens: [' mat', ' floor', ' sofa', ' dog', ' sat'] });
const p = g.softmax(x, -1, { name: 'p' });
const logp = g.log(p, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
const loss = g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });

const steps = createSteps(g, { overview: { ... }, forward: [ ... ] });
mountSteps('#steps', steps);
mountPanels('#values', g, [
  { loss: { sgd: { params: [x], lr: 1, label: '梯度下降一步' }, random: { nodes: [x], range: [-3, 3] }, reset: true, text: '目标类 {{t+1}}，p = {{p[t]:3}}' } },
  { tokens: t, sliders: x, min: -6, max: 6, columns: [p, { node: x, grad: true }], title: 'logits 与目标类' },
], { steps });
```

这里 `x` 是 `(5,)`，label 写 `0`（shape `()`），对应 `nll_loss` 的 `(C,)` 配 `()`。HTML 用 `.vf-stack` 把步骤条和面板竖排。

## 10. 测试页面

本仓库的 e2e 测试用 Playwright，放在 `test/e2e/*.spec.ts`，`npm run e2e` 运行。配置会在随机空闲端口启动 Vite 开发服务器，测试里用相对地址打开页面，例如 `await page.goto('/examples/my-page.html')`。没有安装 Playwright 自带的浏览器时，用环境变量 `VIZFLOW_CHROMIUM` 指定 Chromium 可执行文件：

```sh
VIZFLOW_CHROMIUM=/path/to/chromium npm run e2e -- my-page
```

约定与常用的选择器：

- 页面的 `<head>` 里写 `<link rel="icon" href="data:,">`，否则 favicon 的 404 会出现在控制台错误里。
- 页面脚本最后写 `document.body.dataset.ready = 'true'`，测试先等待 `await expect(page.locator('body')).toHaveAttribute('data-ready', 'true')`。建图或挂载抛错时脚本停止，这个属性不会出现，测试会超时失败。
- 收集控制台错误：`page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); })` 与 `page.on('pageerror', (e) => errors.push(e.message))`，最后断言 `errors` 为空。
- 损失读数：`.vf-loss-value`，文本是 3 位小数，例如 `0.495`。损失面板的说明行：`.vf-sub`。
- 步骤条：当前按钮 `.vf-st[aria-current="step"]`，正文标题 `.vf-caption h3`，正文 `.vf-caption-body`；「上一步」「下一步」按钮的无障碍名称就是这两个词。
- 滑块：`page.getByRole('slider', { name: 'x₁' })`，名称是节点名加下标数字，用 `fill('-1')` 设值。
- 面板按钮：`page.getByRole('button', { name: /梯度下降一步/ })`、`{ name: '重置' }`。
- 类别表：每行一个单选按钮，`page.locator('tbody tr').nth(2).getByRole('radio').check()` 选第 3 类。
- 步骤条按钮 `.vf-st` 的顺序与 `steps.list` 相同：总览、前向各步、反向各步。反向按钮的文字默认以 `∂` 开头。
- 页面可以把图挂到 `window` 上（例如 `window.vizflow = { g, steps }`），测试里用 `page.evaluate` 读数值。

只验证数值时不需要浏览器：在 Vitest 里直接 `import { graph } from '../src/core'` 建图并断言。

## 11. 常见注意事项与报错

| 现象或报错 | 原因与处理 |
|---|---|
| 页面空白，控制台报 CORS 或 module 错误 | 用 `file://` 打开了页面。用 `npm run dev` 或 http server。 |
| `node 'x' already exists` | 节点名重复，比较时忽略空白。 |
| `<op>: unknown option 'xxx'` | 选项对象里写了该运算不支持的键。检查拼写，torch 的位置参数要按位置传。 |
| `node 'h': dimension 'd' is not defined` | shape 里的名字没有写进 `graph({ dims })`。 |
| `node 'x': expected 5 values for shape (5,), got 4` | `values` 的元素个数与 shape 不一致。 |
| `matmul: cannot multiply (1, 4) by (5, 4)` | shape 不匹配。lm_head 的权重存成 `(V, d)`，要乘 `W.T`。 |
| `nll_loss: input (5,) does not match target (1,)` | label 的 shape 不对。单样本 `(C,)` 配 `0`，批量 `(N, C)` 配 `[...]`，见 2.2。 |
| `gather: index (...) must have the same number of dimensions as input (...)` | gather 的 label 维数要与输入相同，单样本写 `[0]`。 |
| `gradient check failed: op 'exp' (node 'u'), input 0 'x' at [2]: backward ... vs numeric ...` | 打开 `check` 时某个运算的 `backward` 与数值导数不一致，通常是自定义运算的 `backward` 写错。 |
| `graph has no loss node` | 挂了 loss 面板，但图里没有元素个数为 1 的运算节点。 |
| 反向步骤一个都没有 | 图里没有损失节点。只演示 softmax 等不含损失的图时这是预期行为。 |
| `backward: the graph has no loss node, so merge and override cannot apply` | 没有损失节点时写了 `merge` 或 `override`。 |
| `override: backward step 6 does not exist (there are 5 steps)` | `override` 的序号超出自动生成的反向步骤数，先用 `'auto'` 看有几步。 |
| `merge: node 'x' is not in any backward step` | 被合并的节点不在 `visible` 中，或在损失之后创建，或是 label。 |
| 反向步骤的节点不对，或者损失不是预期的节点 | 损失之后又创建了标量运算。用 `g.loss(loss)` 指定损失。 |
| `cannot parse template '{{...}}': ...` | 模板写错，见第 4 节。常见的是非标量节点没写下标、名字拼错、对 label 取 `.grad`。 |
| `template at '{{{t+1}} }$' starts with '{{{'; leave a space before the template` | 模板紧贴 LaTeX 花括号，中间加空格。`}}}` 同理。 |
| `cannot parse template '{{b+1}}': 'b' is the current sample index; rename node 'b'` | 图里有名为 `b` 的节点，改名。 |
| `setFocus: 3 is out of range ...` | 样本下标超出可见节点的行数。 |
| `scene3d: node 'X' with shape (2, 3, 4) has more than two dimensions` | 超过二维的节点不能放进 3D 布局。只把二维以内的节点放进 `show` 和布局。 |
| `ui: node 'X' with shape (...) must have at most 2 dimensions` | 面板同样只支持二维以内的节点。 |
| `board: node 'W.T' with shape (...) must be two-dimensional` | `board` 的第二个参数必须是二维矩阵。 |
| `matrix panel: node 'x' must be 2-D, got shape (5,)` | matrix 面板只显示二维张量，一维向量用 tokens 面板或 sliders 面板。 |
| `ui: label 't' needs tokens or classes` | tokens 面板需要类别数，给 label 加 `tokens` 或 `classes`。 |
| `explain: op 'xxx' of node 'y' has no explain definition` | 自定义运算没有注册讲解，却用在了 steps 或 scene3d 中。 |
| `mountScene: container '#stage' not found`、`mountPanels: no element matches '#side'` | 选择器找不到元素。脚本要在容器之后执行，`type="module"` 的脚本默认在 HTML 解析完后执行。 |
| 3D 画布高度为 0 | 容器没有高度。给容器设 `height: var(--vf-stage-height)` 或固定高度。 |
| 卡片上显示 `\mathrm{log p}` 这类写法 | 多字母或含空格的名字没有给 `tex` 和 `elem`。 |
| 拖动滑块后柱子顶到上限 | 刻度在挂载时由初始值决定。用 `scales` 指定刻度。 |
| 某个中间节点没有画出来，也没有反向步骤 | 它没有出现在任何前向步骤的 `show` 里。 |
| 面板不认节点，或运算不存在 | 页面混用了两个来源或版本的入口，见 1.3。 |
