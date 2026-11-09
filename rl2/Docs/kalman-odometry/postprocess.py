import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

INP, OUT = sys.argv[1], sys.argv[2]
EA = 'Microsoft YaHei'
LATIN = 'Segoe UI'
d = Document(INP)

missing = []


def style_or_none(name):
    try:
        return d.styles[name]
    except KeyError:
        missing.append(name)
        return None


def set_style_font(style, size=None, bold=None, color=None, latin=LATIN, ea=EA):
    if style is None:
        return
    f = style.font
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color is not None:
        f.color.rgb = RGBColor.from_string(color)
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:ascii'), latin)
    rFonts.set(qn('w:hAnsi'), latin)
    rFonts.set(qn('w:eastAsia'), ea)
    rFonts.set(qn('w:cs'), latin)


def add_bottom_border(style, sz=14, color='2F6690', space=4):
    pPr = style.element.get_or_add_pPr()
    old = pPr.find(qn('w:pBdr'))
    if old is not None:
        pPr.remove(old)
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), str(sz))
    bottom.set(qn('w:space'), str(space))
    bottom.set(qn('w:color'), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


# ---------- 1. 段落样式 ----------
normal = style_or_none('Normal')
set_style_font(normal, size=10.5, color='1A1A1A')
if normal is not None:
    pf = normal.paragraph_format
    pf.line_spacing = 1.34
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)

for name in ('Body Text', 'First Paragraph'):
    st = style_or_none(name)
    if st is not None:
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.line_spacing = 1.34

compact = style_or_none('Compact')
if compact is not None:
    compact.paragraph_format.space_after = Pt(1)
    compact.paragraph_format.line_spacing = 1.3
    compact.font.size = Pt(10.5)

h1 = style_or_none('Heading 1')
set_style_font(h1, size=17.5, bold=True, color='0F3D63')
if h1 is not None:
    pf = h1.paragraph_format
    pf.page_break_before = True
    pf.keep_with_next = True
    pf.space_before = Pt(0)
    pf.space_after = Pt(14)
    pf.line_spacing = 1.15
    add_bottom_border(h1)

h2 = style_or_none('Heading 2')
set_style_font(h2, size=13.5, bold=True, color='1A4F7A')
if h2 is not None:
    pf = h2.paragraph_format
    pf.keep_with_next = True
    pf.space_before = Pt(14)
    pf.space_after = Pt(5)
    pf.line_spacing = 1.15

h3 = style_or_none('Heading 3')
set_style_font(h3, size=11.5, bold=True, color='23415A')
if h3 is not None:
    pf = h3.paragraph_format
    pf.keep_with_next = True
    pf.space_before = Pt(10)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.15

for name in ('Table Caption', 'Image Caption', 'Caption'):
    st = style_or_none(name)
    if st is not None:
        set_style_font(st, size=9, color='55636F')
        st.paragraph_format.space_after = Pt(6)

# ---------- 2. 页面：A4 + 页边距 ----------
for sec in d.sections:
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    sec.header_distance = Cm(1.1)
    sec.footer_distance = Cm(1.1)
    sec.different_first_page_header_footer = True


# ---------- 3. 封面 ----------
def run_font(run, size, bold=False, color='1A1A1A'):
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), EA)
    rFonts.set(qn('w:ascii'), LATIN)
    rFonts.set(qn('w:hAnsi'), LATIN)


first_par = d.paragraphs[0]


def cover_par(text='', size=10.5, bold=False, color='1A1A1A', before=0, after=0,
              align=WD_ALIGN_PARAGRAPH.LEFT, rule=False):
    p = first_par.insert_paragraph_before()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    if text:
        r = p.add_run(text)
        run_font(r, size, bold, color)
    if rule:
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '18')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), '2F6690')
        pBdr.append(bottom)
        pPr.append(pBdr)
    return p


cover_par(before=150)
cover_par('RoboMaster · STM32H723 · BMI088', size=10, color='4A7FA5', after=20)
cover_par('轮式里程计的卡尔曼滤波', size=28, bold=True, color='0F3D63', after=4)
cover_par('完整公式与讲解', size=16, color='2F6690', after=14)
cover_par(before=6, after=26, rule=True)
cover_par('IMU（BMI088）＋ 轮电机转速 / 编码器', size=11.5, color='3A4750', after=6)
cover_par('两轮差速（含平衡）底盘 · 15 维误差状态卡尔曼滤波', size=11.5, color='3A4750', after=6)
cover_par('2026-10-30', size=10, color='7A8792', before=170)

# ---------- 4. 页脚页码 ----------
def add_page_field(par):
    run = par.add_run()
    fld1 = OxmlElement('w:fldChar'); fld1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = ' PAGE '
    fld2 = OxmlElement('w:fldChar'); fld2.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t'); t.text = '1'
    fld3 = OxmlElement('w:fldChar'); fld3.set(qn('w:fldCharType'), 'end')
    for el in (fld1, it, fld2, t, fld3):
        run._r.append(el)
    run_font(run, 9, False, '7A8792')


sec = d.sections[0]
footer = sec.footer
footer.is_linked_to_previous = False
fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r0 = fp.add_run('— ')
run_font(r0, 9, False, '7A8792')
add_page_field(fp)
r1 = fp.add_run(' —')
run_font(r1, 9, False, '7A8792')

# ---------- 5. 表格：边框、表头底色、字号 ----------
def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def table_borders(table):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn('w:tblBorders'))
    if old is not None:
        tblPr.remove(old)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement('w:' + edge)
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '4')
        e.set(qn('w:space'), '0')
        e.set(qn('w:color'), '9DB4C4')
        borders.append(e)
    tblPr.append(borders)
    oldW = tblPr.find(qn('w:tblW'))
    if oldW is not None:
        tblPr.remove(oldW)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:type'), 'pct')
    tblW.set(qn('w:w'), '5000')
    tblPr.append(tblW)


table_count = 0
for table in d.tables:
    table_count += 1
    table_borders(table)
    for ri, row in enumerate(table.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.size = Pt(9.2)
                    if ri == 0:
                        r.bold = True
            if ri == 0:
                set_cell_shading(cell, 'EAF2F8')

# ---------- 6. 文档属性 ----------
cp = d.core_properties
cp.title = '轮式里程计的卡尔曼滤波：完整公式与讲解'
cp.subject = 'IMU + 轮电机转速/编码器的两轮差速（平衡）底盘 ESKF 里程计'
cp.author = 'DSH'
cp.keywords = '卡尔曼滤波, ESKF, 里程计, BMI088, 编码器, RoboMaster'

d.save(OUT)
print('已保存: ' + OUT)
print('表格数: %d' % table_count)
if missing:
    print('缺少样式: ' + ', '.join(missing))
