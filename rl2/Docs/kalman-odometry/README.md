# 轮式里程计卡尔曼滤波文档

成品：`../轮式里程计卡尔曼滤波.pdf`（A4 / 21 页 / 136 个独立公式 + 170 个行内公式）

## 文件说明

| 文件 | 用途 |
|---|---|
| `content.md` | 正文源文件（Markdown + LaTeX 公式） |
| `postprocess.py` | 排版：A4 页边距、封面、页脚页码、标题样式、表格边框 |
| `make-pdf.ps1` | 一键构建：pandoc → docx（公式转 Word 原生 OMML）→ LibreOffice Kit → PDF |

## 重新生成

1. 将 pandoc 便携版解压到 `build/pandoc/pandoc-<版本>/pandoc.exe`（当前为 3.13，
   Windows x86_64 zip）。
2. 运行 `make-pdf.ps1`。

公式走 OMML 而非图片，因此 PDF 中的公式是矢量文本，可选中、可搜索；
改公式只需编辑 `content.md` 里的 `$...$` / `$$...$$` 后重新构建。

## 公式写法注意事项（已踩过的坑）

- 正体函数名用 `\text{diag}` 或 `\operatorname{tr}`，**不要用 `\mathrm{}`**
  （LibreOffice 导入 OMML 时会渲染成「d i a g」）。
- 竖线分隔用 `\vert` 或 `|`，`\mid` 会被渲染成斜杠。
- 分式用 `\frac`，`\tfrac` 会被渲染成线性的 `1/2`。
- 公式内不要出现中文字符（OMML 转换会失败），中文放到公式外。
- 矩阵行距选项 `\\[2pt]` 不被 texmath 支持，改用 `\\`。
