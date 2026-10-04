from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "docx" / "Технические_требования_к_данным_Pregnancy_State_Platform.docx"

FONT = "Liberation Sans"
BLACK = "000000"
NAVY = "18344B"
MID_BLUE = "2E607D"
PALE_BLUE = "EAF2F7"
PALE_GRAY = "F4F6F7"
LIGHT_GRAY = "D9D9D9"
WHITE = "FFFFFF"
MUTED = "4D5B66"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color: str = LIGHT_GRAY, size: str = "6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        node = tc_borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            tc_borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def set_cell_margins(cell, top=95, start=110, bottom=95, end=110) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    # The bundled LibreOffice renderer can place repeated Word table headers
    # above the printable area on continuation pages. Keeping the first header
    # only avoids clipped rows in the reviewed deliverable.
    return None


def set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def set_column_width(cell, width_inches: float) -> None:
    cell.width = Inches(width_inches)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_font(run, size: float | None = None, bold: bool | None = None, color: str = BLACK) -> None:
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    run.font.color.rgb = RGBColor.from_string(color)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_para_spacing(paragraph, before=0, after=5, line=1.08, keep_next=False) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    fmt.keep_with_next = keep_next


def remove_paragraph_border(paragraph_or_style) -> None:
    element = paragraph_or_style._element
    p_pr = element.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is not None:
        p_pr.remove(p_bdr)


def add_text(paragraph, text: str, bold: bool = False, color: str = BLACK, size: float | None = None):
    run = paragraph.add_run(text)
    set_font(run, size=size, bold=bold, color=color)
    return run


def add_hyperlink(paragraph, label: str, url: str):
    part = paragraph.part
    rel_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), FONT)
    r_fonts.set(qn("w:hAnsi"), FONT)
    r_fonts.set(qn("w:eastAsia"), FONT)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), MID_BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.extend([r_fonts, color, underline])
    text = OxmlElement("w:t")
    text.text = label
    new_run.extend([r_pr, text])
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    set_font(run, size=8.5, color=MUTED)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])


def add_paragraph(doc: Document, text: str = "", bold_lead: str | None = None, after=5, before=0):
    p = doc.add_paragraph()
    set_para_spacing(p, before=before, after=after)
    if bold_lead and text.startswith(bold_lead):
        add_text(p, bold_lead, bold=True)
        add_text(p, text[len(bold_lead):])
    else:
        add_text(p, text)
    return p


def add_bullets(doc: Document, items: Iterable[str], level: int = 0):
    for item in items:
        p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        set_para_spacing(p, after=2.5, line=1.05)
        add_text(p, item)


def add_numbered(doc: Document, items: Iterable[str]):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        set_para_spacing(p, after=3, line=1.06)
        add_text(p, item)


def add_table(
    doc: Document,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[float],
    font_size: float = 8.8,
    header_fill: str = NAVY,
):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    set_row_cant_split(hdr)
    for idx, (cell, header) in enumerate(zip(hdr.cells, headers)):
        set_column_width(cell, widths[idx])
        set_cell_shading(cell, header_fill)
        set_cell_border(cell)
        set_cell_margins(cell, top=120, bottom=120)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p, after=0, line=1.0)
        add_text(p, header, bold=True, color=WHITE, size=9.0)

    for row_idx, data in enumerate(rows):
        row = table.add_row()
        set_row_cant_split(row)
        fill = WHITE if row_idx % 2 == 0 else PALE_BLUE
        for idx, (cell, value) in enumerate(zip(row.cells, data)):
            set_column_width(cell, widths[idx])
            set_cell_shading(cell, fill)
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx == 0 and len(headers) > 2 else WD_ALIGN_PARAGRAPH.LEFT
            set_para_spacing(p, after=0, line=1.02)
            add_text(p, str(value), bold=(idx == 0), size=font_size)
    spacer = doc.add_paragraph()
    set_para_spacing(spacer, after=2, line=1.0)
    return table


def add_heading(doc: Document, text: str, level: int = 1, page_break_before: bool = False):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = False
    p.paragraph_format.page_break_before = page_break_before
    if page_break_before:
        p.paragraph_format.space_before = Pt(24)
    return p


def add_section_break(doc: Document):
    # Major sections are allowed to flow. Forced breaks after long tables create
    # large blank areas and make the request harder to print and review.
    return None


def add_source(doc: Document, title: str, details: str, url: str | None = None):
    p = doc.add_paragraph()
    set_para_spacing(p, after=4, line=1.06)
    add_text(p, title, bold=True)
    add_text(p, f". {details}")
    if url:
        add_text(p, " ")
        add_hyperlink(p, "Официальный источник", url)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(10.7)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08

    title = styles["Title"]
    title.font.name = FONT
    title._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    title._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    title._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    title.font.size = Pt(25)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(BLACK)
    title.paragraph_format.space_after = Pt(14)
    title.paragraph_format.line_spacing = 1.0
    remove_paragraph_border(title)

    if "Subtitle" in styles:
        subtitle = styles["Subtitle"]
        subtitle.font.name = FONT
        subtitle._element.rPr.rFonts.set(qn("w:ascii"), FONT)
        subtitle._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        subtitle._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        subtitle.font.size = Pt(12.5)
        subtitle.font.color.rgb = RGBColor.from_string(BLACK)
        subtitle.paragraph_format.space_after = Pt(18)

    for name, size, before, after in (
        ("Heading 1", 16.5, 16, 8),
        ("Heading 2", 13.0, 12, 6),
        ("Heading 3", 11.2, 9, 4),
    ):
        style = styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLACK)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = False

    for name in ("List Bullet", "List Bullet 2", "List Number"):
        style = styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(10.4)
        style.font.color.rgb = RGBColor.from_string(BLACK)

    if "Table Text" not in styles:
        table_style = styles.add_style("Table Text", WD_STYLE_TYPE.PARAGRAPH)
        table_style.font.name = FONT
        table_style.font.size = Pt(8.8)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.28)

    footer = section.footer
    p = footer.paragraphs[0]
    set_para_spacing(p, after=0, line=1.0)
    add_page_number(p)


def build_document() -> Document:
    doc = Document()
    configure_styles(doc)
    configure_document(doc)

    props = doc.core_properties
    props.title = "Технические требования к обезличенному продольному набору данных по беременности родам и неонатальным исходам"
    props.subject = "Запрос данных для разработки и проверки исследовательских моделей Pregnancy State Platform"
    props.author = "Pregnancy State Platform"
    props.keywords = "беременность, продольные данные, машинное обучение, КТГ, EHG, преэклампсия, преждевременные роды"

    # Cover
    p = doc.add_paragraph()
    set_para_spacing(p, before=20, after=24)
    add_text(p, "PREGNANCY STATE PLATFORM", bold=True, color=MID_BLUE, size=9.5)

    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    remove_paragraph_border(p)
    add_text(p, "Технические требования к обезличенному продольному набору данных по беременности родам и неонатальным исходам", bold=True, size=25)

    p = doc.add_paragraph(style="Subtitle")
    add_text(p, "Запрос для разработки и проверки исследовательских моделей риска", size=12.5)

    for label, value in (
        ("Версия", "1.1"),
        ("Дата", "23 сентября 2026 года"),
        ("Назначение", "согласование состава выгрузки с медицинской организацией"),
        ("Режим", "обезличенные или псевдонимизированные данные в защищенном контуре"),
    ):
        p = doc.add_paragraph()
        set_para_spacing(p, after=4)
        add_text(p, f"{label}  ", bold=True, color=MUTED)
        add_text(p, value)

    p = doc.add_paragraph()
    set_para_spacing(p, before=22, after=6)
    add_text(p, "Главный запрос", bold=True, size=11.5)
    add_text(
        p,
        "  предоставить последовательную когорту беременностей с полной временной связью между наблюдениями, вмешательствами и исходами. "
        "В первую очередь нужны данные для моделей преэклампсии и гипертензивных нарушений, преждевременных родов, КТГ и EHG. "
        "Остальные блоки нужны для будущих карточек риска и формирования единого набора данных стартапа.",
    )

    add_paragraph(
        doc,
        "Выдача может быть поэтапной. Каждый этап должен оставаться связным: стабильные идентификаторы женщины, беременности и плода, "
        "хронология, единицы измерения, источник, версия выгрузки и исходы сохраняются во всех пакетах.",
        after=0,
    )

    doc.add_page_break()

    add_heading(doc, "Содержание", 1)
    toc_items = [
        "1 Цель и границы запроса",
        "2 Когорта и принцип отбора",
        "3 Поэтапная передача данных",
        "4 Единая модель данных",
        "5 Общий клинический набор",
        "6 Преэклампсия и гипертензивные нарушения",
        "7 Преждевременные роды",
        "8 КТГ и EHG",
        "9 ИЦН",
        "10 Задержка роста плода",
        "11 Гестационный сахарный диабет",
        "12 Венозные тромбоэмболические осложнения",
        "13 Материнские и неонатальные исходы",
        "14 Дополнительный слой для SSM",
        "15 Форматы и структура поставки",
        "16 Обезличивание и правовой контур",
        "17 Контроль качества и приемка",
        "18 Пакет первого согласования",
        "Приложение А Логические таблицы",
        "Приложение Б Справочники и единицы",
        "Приложение В Источники",
    ]
    add_bullets(doc, toc_items)

    add_heading(doc, "1 Цель и границы запроса", 1)
    add_paragraph(
        doc,
        "Документ задает максимальный состав обезличенного набора данных, который позволит последовательно разработать легкие интерпретируемые модели, "
        "оценить модели временных сигналов и позднее перейти к продольной SSM модели. Набор должен отражать реальную помощь, включая нормальные результаты, "
        "отрицательные исходы, пропуски, изменения лечения и исправления записей.",
    )
    add_paragraph(
        doc,
        "Перечень не предписывает назначать пациенткам дополнительные анализы или исследования. Передаются только фактически выполненные исследования и "
        "зафиксированные сведения. Если показатель в организации не собирается, это отмечается в карте доступности данных.",
        bold_lead="Перечень не предписывает назначать пациенткам дополнительные анализы или исследования.",
    )
    add_paragraph(
        doc,
        "Модели являются исследовательскими. Выгрузка сама по себе не меняет клиническую тактику. Определения исходов и действующие пороги утверждаются "
        "совместно с клиническими экспертами медицинской организации по актуальным клиническим рекомендациям и локальным регламентам.",
    )

    add_table(
        doc,
        ["Приоритет", "Задача", "Назначение данных", "Первый модельный уровень"],
        [
            ["P0", "Преэклампсия и гипертензивные нарушения", "Динамический риск на ближайшие недели и до родов", "Регуляризованная логистика и landmark модель"],
            ["P0", "Преждевременные роды", "Риск родов до срока и время до события", "Логистика и discrete time survival"],
            ["P0", "КТГ и EHG", "Качество сигнала, морфология и связь с исходом", "Ручные признаки, логистика, затем CNN или TCN"],
            ["P1", "Гестационный сахарный диабет", "Ранний риск до ОГТТ", "Простой табличный baseline"],
            ["P2", "ИЦН и задержка роста плода", "Накопление серийных данных и достоверных фенотипов", "После проверки полноты локальной когорты"],
            ["Правило", "Венозные тромбоэмболические осложнения", "Версионируемая карточка факторов риска и исходов", "Rule only до достаточной локальной когорты"],
        ],
        [0.55, 1.7, 2.75, 1.65],
        font_size=8.5,
    )

    add_heading(doc, "2 Когорта и принцип отбора", 1)
    add_paragraph(
        doc,
        "Предпочтителен полный последовательный регистр всех беременностей, наблюдавшихся или завершившихся в организации за пять и более полных календарных лет. "
        "Точный период определяется после инвентаризации. Для каждой беременности нужен исход либо явная причина неполного наблюдения.",
    )
    add_bullets(doc, [
        "Включать нормальные беременности и все отрицательные случаи. Выборка только осложнений непригодна для оценки абсолютного риска и калибровки.",
        "Сохранять все беременности одной женщины под одним устойчивым псевдонимом. При разделении train и test они остаются в одной части данных.",
        "Сохранять отдельный идентификатор каждого плода. Данные двойни или тройни нельзя хранить в одной неразличимой строке.",
        "Включать амбулаторные визиты, госпитализации, роды, ранний послеродовый период и данные новорожденного, даже если они находятся в разных системах.",
        "Не исключать переведенных пациенток и неполные случаи. Для них указывать источник поступления, перевод, потерю наблюдения и доступность исхода.",
        "Для редких исходов допустим отдельный обогащенный набор случаев, но он не заменяет последовательную популяционную когорту.",
    ])

    add_heading(doc, "Предварительная инвентаризация", 2)
    add_table(
        doc,
        ["Показатель", "Что предоставить", "Разрез"],
        [
            ["Объем", "Число женщин, беременностей, плодов, родов и новорожденных", "По календарному году и подразделению"],
            ["Исходы", "Число случаев каждого целевого фенотипа и число событий с неизвестным статусом", "По году, сроку и способу подтверждения"],
            ["Наблюдения", "Медиана и квартили числа визитов, лабораторий, УЗИ и госпитализаций на беременность", "По году и группе риска"],
            ["Сигналы", "Число записей КТГ и EHG, суммарные минуты, доля связанных с родами и неонатальными исходами", "По устройству и году"],
            ["Системы", "МИС, ЛИС, РИС или PACS, архивы приборов, аптечная, операционная, реанимационная и неонатальная системы", "Название системы, версия и периоды работы"],
            ["Изменения", "Миграции, смены приборов, единиц, референсов, шаблонов и локальных протоколов", "Дата начала и окончания версии"],
        ],
        [1.15, 4.0, 1.5],
    )

    add_heading(doc, "3 Поэтапная передача данных", 1, page_break_before=True)
    add_paragraph(
        doc,
        "Целевой набор остается единым, но передается пакетами. Каждый пакет имеет идентификатор снимка, дату формирования, версию алгоритма выгрузки, "
        "описание фильтров, контрольные суммы, количество строк и перечень исправлений относительно предыдущей версии.",
    )
    add_table(
        doc,
        ["Этап", "Содержание", "Цель", "Критерий завершения"],
        [
            ["0", "Карта источников, словарь полей, статистика доступности и 200–500 последовательных беременностей", "Проверить связность и обезличивание", "Совместно разобраны источники, единицы, пропуски и 20–30 тестовых историй"],
            ["1", "Структурированная когорта для преэклампсии и преждевременных родов с исходами", "Построить легкие baseline модели", "Хронология, labels и point in time признаки воспроизводимы"],
            ["2", "Raw КТГ и EHG с метаданными, контекстом родов и неонатальными исходами", "Signal baseline и контроль качества", "Записи декодируются, синхронизируются и не смешиваются между пациентками"],
            ["3", "ИЦН, FGR, GDM и ВТЭО, серийное УЗИ, допплер, вмешательства", "Карточки сбора и будущие модели", "Поля связаны с беременностью и плодом, исходы подтверждены"],
            ["4", "Тексты, документы, DICOM, назначения и введения препаратов, план наблюдения", "Подготовка мультимодальной SSM модели", "Сохранены порядок событий, provenance, маски доступности и причины пропусков"],
            ["Дельты", "Новые случаи, исправления и поздние исходы после каждого полного снимка", "Обновление без потери истории", "Исправления приходят новыми версиями, а не молча заменяют прошлое"],
        ],
        [0.7, 2.75, 1.55, 1.65],
        font_size=8.3,
    )

    add_heading(doc, "4 Единая модель данных", 1)
    add_paragraph(
        doc,
        "Основная иерархия набора: женщина, беременность, плод, контакт с медицинской организацией, клиническое событие и исход. "
        "Связь должна сохраняться при любой поэтапной поставке.",
    )
    add_table(
        doc,
        ["Сущность", "Обязательный идентификатор", "Минимальное содержание"],
        [
            ["Женщина", "person_id", "Устойчивый псевдоним, год рождения или возраст на событие, разрешенные демографические признаки"],
            ["Беременность", "pregnancy_id", "Начало эпизода, датировка, ПДР, способ наступления, плодность, исход и полнота наблюдения"],
            ["Плод", "fetus_id", "Метка плода, хориальность, амниальность, серийные УЗИ и индивидуальный исход"],
            ["Контакт", "encounter_id", "Дата и время, тип контакта, подразделение, амбулаторно или стационарно, причина обращения"],
            ["Событие", "event_id", "Показатель, значение, единица, метод, время наблюдения, время доступности и источник"],
            ["Документ", "document_id", "Тип, дата создания и подписания, авторская роль, связь с событиями, версия и обезличенный текст"],
            ["Исход", "outcome_id", "Определение, дата или гестационный срок, источник, статус подтверждения и версия adjudication"],
        ],
        [1.0, 1.45, 4.2],
    )

    add_heading(doc, "Время и датировка беременности", 2)
    add_bullets(doc, [
        "observed_at — когда измерение выполнено, симптом возник, препарат введен или образец получен.",
        "available_at — когда результат был подписан или стал доступен врачу и информационной системе.",
        "ordered_at, specimen_at, resulted_at и verified_at передаются отдельно, если система их хранит.",
        "ga_days — срок беременности в полных днях на момент события. Недельная запись не заменяет точное число дней.",
        "dating_method, dating_anchor, EDD, версия датировки, уверенность и время изменения сохраняются. Новая ПДР не перезаписывает старую.",
        "Для ЭКО передаются тип переноса, дата переноса и возраст эмбриона; для естественной беременности — ПМ и данные раннего УЗИ.",
        "Если календарные даты сдвигаются при обезличивании, сдвиг должен быть постоянным внутри женщины и сохранять интервалы, срок беременности и порядок событий.",
    ])

    add_heading(doc, "Атрибуты каждого клинического факта", 2)
    add_table(
        doc,
        ["Группа", "Обязательные поля", "Пояснение"],
        [
            ["Значение", "Исходное значение, единица, текстовое представление и нормализованное значение", "Ноль, отрицательный результат и отсутствие данных различаются"],
            ["Метод", "Материал, методика, анализатор или прибор, модель, программная версия", "Нужно для оценки смены технологии и внешней переносимости"],
            ["Референс", "Нижняя и верхняя границы, текстовый флаг, лаборатория", "Референсы сохраняются такими, какими были на дату результата"],
            ["Качество", "Статус, признак отмены, гемолиз, повтор, техническая ошибка, signal loss", "Исправленный факт получает новую версию"],
            ["Происхождение", "Система, таблица или документ, локальный код, document_id и фрагмент источника", "Позволяет проверить любой модельный признак"],
            ["Доступность", "Причина отсутствия или неполноты", "not_due, clinician_decision, patient_declined, unavailable, lost_to_followup, data_error"],
        ],
        [1.0, 3.0, 2.65],
    )

    add_heading(doc, "5 Общий клинический набор", 1)
    add_paragraph(
        doc,
        "Этот блок нужен всем моделям и предотвращает повторные несовместимые выгрузки. Поля передаются как датированные события, а не как одна итоговая строка на беременность.",
    )
    add_table(
        doc,
        ["Блок", "Что передать", "Обязательные атрибуты"],
        [
            ["Демография", "Возраст на дату события, год рождения, гражданство и регион проживания в допустимой укрупненной форме, язык при наличии", "Не передавать адрес, телефон, ФИО, СНИЛС, полис и точные геокоординаты"],
            ["Антропометрия", "Рост, добеременный и текущий вес, ИМТ как производное, прибавка массы", "Дата, срок, единица, источник, условия измерения"],
            ["Общий анамнез", "Хроническая гипертензия, болезни почек, печени, сердца, сахарный диабет, аутоиммунные состояния, АФС, тромбофилия, заболевания щитовидной железы, эпилепсия", "Дата начала, активность, источник подтверждения, МКБ код и свободный текст при наличии"],
            ["Репродуктивный анамнез", "Число беременностей и родов, исход и точный срок каждой беременности, выкидыши, мертворождение, внематочная беременность, осложнения", "Предыдущие беременности хранятся отдельными эпизодами, если доступны"],
            ["Текущая беременность", "ПМ, ПДР, способ датировки, естественная или ВРТ, плодность, хориальность и амниальность, резус и группа крови", "Версия и дата каждого изменения"],
            ["Поведение и экспозиции", "Курение, никотин, алкоголь, психоактивные вещества, условия труда и значимые вредности", "Период, интенсивность, источник; чувствительные поля только при правовом основании"],
            ["Витальные показатели", "АД, пульс, температура, SpO2, частота дыхания, диурез, отеки", "Отдельное событие для каждого измерения и повтора"],
            ["Диагнозы", "Все коды МКБ 10, текст диагноза, основной или сопутствующий, предполагаемый или подтвержденный", "Дата появления, изменения и закрытия, авторская роль, источник"],
            ["Лекарства", "МНН, торговое наименование, форма, доза, путь, кратность, назначение, отмена и фактическое введение", "Назначение и введение хранятся раздельно; указывается показание и приверженность, если доступна"],
            ["Процедуры", "Осмотры, операции, анестезия, трансфузии, реанимационные мероприятия", "Время начала и окончания, показание, результат, осложнение"],
            ["Контакты", "Плановые и внеплановые визиты, вызовы, госпитализации, переводы, выписки", "Подразделение, тип, причина, диагноз при поступлении и выписке"],
            ["План наблюдения", "Назначено, ожидается, выполнено, просрочено, отменено, отказ или перенос", "Плановое окно, условие назначения, причина невыполнения"],
        ],
        [1.2, 3.2, 2.25],
        font_size=8.2,
    )

    add_heading(doc, "Источники медицинской документации", 2)
    add_table(
        doc,
        ["Источник", "Что запросить", "Как связать"],
        [
            ["Женская консультация", "Амбулаторная карта, индивидуальная карта беременной и родильницы, обменная карта, протоколы осмотров и направления", "person_id, pregnancy_id, encounter_id, дата создания и подписания"],
            ["Стационар", "История болезни, история родов, партограмма, дневники, эпикризы, протоколы консилиумов и перевода", "pregnancy_id, encounter_id, подразделение, начало и окончание госпитализации"],
            ["Назначения", "Листы назначений, фактические введения, инфузии, листы наблюдения, аптечные выдачи", "medication_order_id, administration_id, точное время и статус"],
            ["Операционная и анестезия", "Протокол операции, анестезиологическая карта, нейроаксиальные процедуры, кровопотеря и трансфузия", "procedure_id, start_time, end_time и показание"],
            ["ЛИС", "Исходные лабораторные протоколы, accession, материал, анализатор, референсы, отмены и исправления", "event_id, specimen_at, resulted_at, verified_at"],
            ["УЗИ и допплер", "Протоколы, таблицы измерений, заключения, ключевые изображения и DICOM при наличии", "study_id, pregnancy_id, fetus_id, device_id"],
            ["КТГ и приборные архивы", "Raw файлы, отчеты, автоматический анализ, врачебная интерпретация и журнал прибора", "recording_id, fetus_id, start_time, end_time"],
            ["Неонатология", "История развития новорожденного, реанимационный лист, ОРИТН, лаборатории, процедуры и выписка", "newborn_id, fetus_id, birth_time"],
            ["Патология", "Исследование плаценты, оболочек и пуповины, макроописание и гистология", "specimen_id, pregnancy_id, fetus_id и document_id"],
            ["Повторные обращения", "Послеродовые госпитализации, обращения матери до 42 дней и исходы ребенка до 28 дней или выписки", "person_id, pregnancy_id, newborn_id и encounter_id"],
        ],
        [1.45, 3.65, 1.55],
        font_size=8.0,
    )

    add_heading(doc, "Базовые лабораторные исследования", 2)
    add_paragraph(
        doc,
        "Запрашивается полный массив ЛИС по включенным беременностям, а не только перечисленные ниже показатели. Таблица является семантическим минимумом для приемки, "
        "но не белым списком: сохраняются все фактически выполненные компоненты панелей, включая нормальные и отрицательные результаты, повторы, отмены и исправления. "
        "Уровень A означает ядро общей когорты, B — профильные исследования приоритетных рисков, C — условные и редкие исследования, передаваемые только если они были назначены в реальной помощи.",
    )
    add_paragraph(
        doc,
        "Для каждого анализа нужны: order_id, accession_id и specimen_id при наличии; локальный код и полное название аналита; материал и место забора; ordered_at, specimen_at, resulted_at, verified_at и available_at; "
        "исходное значение и единица; предел обнаружения и знак «меньше/больше»; референс на дату; метод, анализатор, реагент или партия при наличии; флаги гемолиза, липемии, иктеричности, разведения, повтора, отмены и коррекции. "
        "Компоненты панели передаются отдельными строками и связываются общим panel_id. Отсутствующий анализ не кодируется как нормальный.",
    )
    add_table(
        doc,
        ["Уровень и группа", "Показатели", "Особые требования"],
        [
            ["A Гематология", "Гемоглобин, гематокрит, эритроциты, MCV, MCH, MCHC, RDW, лейкоциты; формула в абсолютных и относительных значениях; тромбоциты, MPV", "Исходные единицы, анализатор, референс, морфологические и критические флаги"],
            ["B/C Анемия и морфология", "Ферритин; железо, трансферрин, ОЖСС или НЖСС, насыщение трансферрина; ретикулоциты, витамин B12, фолат; ручной мазок", "Ферритин — при выполнении; остальные показатели условные. Указать воспалительный контекст и метод"],
            ["A Почки и электролиты", "Креатинин, мочевина, мочевая кислота; натрий, калий, хлор, кальций, магний; расчетная СКФ или клиренс, если рассчитаны", "Первичные значения не заменять расчетами; сохранить формулу, версию и ограничения расчета"],
            ["A/B Печень, белок и гемолиз", "Общий белок, альбумин, АСТ, АЛТ, ЛДГ, билирубин общий, прямой и непрямой, ЩФ, ГГТ; при наличии гаптоглобин, свободный гемоглобин, шизоциты и сфероциты", "Компоненты и фракции отдельно; учитывать физиологические изменения беременности"],
            ["A/B Гемостаз", "АЧТВ, ПВ в секундах и процентах или ПТИ, МНО, фибриноген, тромбиновое время; при наличии D-димер, антитромбин, анти-Xa и факторы свертывания", "Для D-димера обязательны DDU или FEU; для анти-Xa — препарат, доза и время от введения"],
            ["A/B Углеводный обмен", "Глюкоза венозной плазмы натощак и случайная; ОГТТ 0, 60 и 120 минут; HbA1c при выполнении; кетоны; капиллярный самоконтроль и CGM отдельно", "Материал, натощак, лабораторный или портативный метод, фактическое время относительно еды и нагрузки"],
            ["A/C Щитовидная железа", "ТТГ и АТ-ТПО на первом визите; свободный T4, свободный T3 и иные тесты при наличии показаний", "Не смешивать единицы и методы; передать гестационный срок и референс конкретной лаборатории"],
            ["A/B Моча и протеинурия", "Общий анализ: цвет, прозрачность, pH, удельный вес, белок, глюкоза, кетоны, кровь, эритроциты, лейкоциты, нитриты, бактерии, цилиндры и осадок; белок и альбумин количественно, креатинин мочи, ACR, PCR, суточный белок", "Тип порции, способ получения, тест-полоска или лаборатория; для суточной мочи — объем, начало, конец и полнота сбора"],
            ["A Инфекционный скрининг", "ВИЧ 1/2 Ag/Ab, HBsAg, anti-HCV, антитела к Treponema pallidum; IgG и IgM к краснухе; микроскопия влагалищного мазка; посев мочи; скрининг S. agalactiae", "Передать все отрицательные и подтверждающие этапы, материал, метод, срок беременности и лечение до забора"],
            ["B/C Инфекция и микробиология", "ПЦР и культура влагалищного, цервикального, ректального отделяемого, мочи, крови, плаценты и околоплодных вод; идентификация и чувствительность; TORCH и ИППП только если выполнены", "Организм, количественный результат или КОЕ, панель мишеней, антибиотикограмма, время антибиотика относительно забора"],
            ["A Иммуногематология", "AB0, RhD, скрининг антител к эритроцитам, идентификация, титр; прямая или непрямая проба Кумбса при выполнении", "Дата, метод, специфичность, критический комментарий и введение иммуноглобулина anti-D"],
            ["A/B Пренатальный скрининг", "Свободная beta-субъединица ХГЧ, PAPP-A; PlGF при включении в локальный алгоритм; AFP, неконъюгированный эстриол, ингибин A, НИПТ и инвазивная генетика при выполнении", "Исходные концентрации и MoM, fetal fraction и контроль качества, программа, версия алгоритма, рассчитанные риски"],
            ["C АФС и тромбофилия", "Волчаночный антикоагулянт и его этапы, антикардиолипин IgG или IgM, anti-beta2-GPI IgG или IgM; антитромбин, протеин C и S; Factor V Leiden, протромбин G20210A и иные выполненные тесты", "Только фактически назначенные тесты; дата, беременность, острый тромбоз и антикоагулянтная терапия влияют на интерпретацию"],
        ],
        [1.35, 3.55, 1.75],
        font_size=7.75,
    )

    add_heading(doc, "6 Преэклампсия и гипертензивные нарушения", 1, page_break_before=True)
    add_paragraph(
        doc,
        "Нужны три разных набора меток: анамнестический риск на первом визите, динамический прогноз на фиксированную дату и подтвержденный клинический исход. "
        "Хроническая гипертензия, гестационная гипертензия, преэклампсия, преэклампсия на фоне хронической гипертензии, HELLP и эклампсия не объединяются в один непрозрачный флаг.",
    )

    add_heading(doc, "Анамнез и факторы риска", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Временная привязка"],
        [
            ["Предыдущие беременности", "Преэклампсия, гипертензия, HELLP, эклампсия, ЗРП, отслойка плаценты, досрочное родоразрешение, срок и тяжесть", "Для каждого предыдущего эпизода отдельно"],
            ["Заболевания", "Хроническая АГ, болезни почек, диабет, СКВ, АФС, сердечно сосудистые заболевания, тромбофилия", "Диагноз должен предшествовать index time или иметь дату начала"],
            ["Текущая беременность", "Первая беременность, многоплодие, ВРТ, донорская яйцеклетка, интервал между беременностями, ранний ИМТ", "На первом контакте и при уточнении"],
            ["Семейный анамнез", "Преэклампсия, ранняя АГ и сердечно сосудистые события у родственников", "Источник и степень родства"],
            ["Профилактика", "Ацетилсалициловая кислота и кальций при наличии назначения", "Показание, назначение, фактический прием, доза, начало, отмена и приверженность"],
        ],
        [1.25, 3.75, 1.65],
    )

    add_heading(doc, "Динамика состояния", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Обязательные метаданные"],
        [
            ["Артериальное давление", "САД и ДАД для каждого измерения, домашние измерения и суточное мониторирование при наличии", "Рука, положение, размер манжеты, прибор, повтор, порядок, дата и время, клинический контекст"],
            ["Жизненные показатели", "Пульс, SpO2, частота дыхания, температура, вес, динамика массы, диурез и водный баланс", "Наблюдение и время доступности"],
            ["Симптомы", "Головная боль, нарушения зрения, боль в эпигастрии или правом подреберье, тошнота, рвота, одышка, отеки, судороги, снижение диуреза", "Начало, выраженность, длительность, отрицание симптома и источник"],
            ["Осмотр", "Отеки, неврологический статус, рефлексы и клонус при наличии, признаки отека легких", "Дата, авторская роль и свободный комментарий"],
            ["Госпитализация", "Причина, отделение, уровень помощи, перевод в ПИТ или ОРИТ", "Поступление, перевод, выписка и итоговый диагноз"],
        ],
        [1.25, 3.6, 1.8],
    )

    add_heading(doc, "Лабораторные и инструментальные данные", 2)
    add_table(
        doc,
        ["Группа", "Показатели", "Особые требования"],
        [
            ["Протеинурия", "Белок мочи качественно и количественно, суточная протеинурия, отношение белок к креатинину, альбумин к креатинину", "Тип образца, объем суточной мочи, г на литр, г в сутки, мг на ммоль или мг на г, метод"],
            ["Кровь", "Гемоглобин, гематокрит, тромбоциты; мазок с морфологией эритроцитов, тромбоцитов и лейкоцитов, шистоциты и сфероциты", "Повторы хранить отдельно; абсолютные значения и проценты, анализатор, ручной просмотр и флаги"],
            ["Почки", "Креатинин, мочевина, мочевая кислота, электролиты, расчетная СКФ с формулой при наличии", "Первичное значение не заменять расчетным показателем"],
            ["Печень и гемолиз", "АСТ, АЛТ, ЛДГ, билирубин общий, прямой и непрямой, гаптоглобин, свободный гемоглобин в крови или моче, ретикулоциты, альбумин и общий белок", "Основное ядро — биохимия из КР; расширенные маркеры передаются при выполнении"],
            ["Гемостаз", "АЧТВ, ПВ, ПТИ, МНО, фибриноген; тромбиновое время, D-димер и антитромбин при выполнении", "Метод, единицы и тип D-димера DDU или FEU; все повторы до и после лечения"],
            ["ТМА, условно", "Активность ADAMTS13 и ингибитор, прямая проба Кумбса, C3, C4, CH50, генетические и иные тесты дифференциальной диагностики ТТП или аГУС", "Не рутинный набор: передать только назначенные тесты вместе с показанием, временем и терапией до забора"],
            ["Ангиогенные маркеры", "PlGF, sFlt 1 и рассчитанное отношение", "При наличии; анализатор, реагент, срок, единицы и версия порога"],
            ["Первый скрининг", "КТР, ТВП, PAPP-A, свободный beta-ХГЧ, PlGF, средний PI маточных артерий, среднее АД и рассчитанные риски ПЭ или ЗРП", "Исходные значения и MoM, программа, версия и дата расчета, прибор, срок, качество и пороги риска"],
            ["УЗИ и допплер", "Маточные артерии, артерия пуповины, средняя мозговая артерия, CPR, венозный проток, воды и рост", "Привязка к fetus_id, прибор, оператор, индексы и качество"],
            ["Состояние плода", "КТГ, компьютеризированная КТГ и краткосрочная вариабельность при наличии, биофизический профиль, движения плода", "Raw запись, версия алгоритма, интерпретация и время решения"],
            ["Неврология, условно", "КТ или МРТ головного мозга и иные исследования при атипичных или резистентных судорогах и очаговой симптоматике", "Изображения или study_id, время, показание и заключение"],
        ],
        [1.2, 3.45, 2.0],
        font_size=8.2,
    )

    add_heading(doc, "Лечение и исходы", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Почему важно для модели"],
        [
            ["Антигипертензивная терапия", "МНН, доза, путь, назначение, фактическое введение, изменение и причина", "Лечение меняет последующие давление и риск"],
            ["Магния сульфат", "Показание, болюс, инфузия, время начала и окончания, контроль", "Отдельная временная экспозиция"],
            ["Кортикостероиды", "Препарат, доза, полный или неполный курс, даты", "Связь с угрозой досрочного родоразрешения и неонатальным исходом"],
            ["Родоразрешение", "Дата и время, срок, способ, срочность, показания, состояние шейки, индукция", "Медицински показанное преждевременное родоразрешение не равно spontaneous PTB"],
            ["Материнские исходы", "HELLP, эклампсия, отслойка плаценты, ОПП, повреждение печени, отек легких, инсульт, трансфузия, ОРИТ, смерть", "Дата начала, критерии, подтверждение и разрешение"],
            ["Исходы плода", "ЗРП, антенатальная гибель, состояние при рождении, газы пуповинной крови, Апгар, реанимация и ОРИТН", "На каждого плода отдельно"],
        ],
        [1.3, 3.55, 1.8],
        font_size=8.2,
    )

    add_heading(doc, "Контракт меток", 2)
    add_bullets(doc, [
        "Точный фенотип, версия определения и исходные критерии, а не только выписной МКБ код.",
        "Дата первого выполнения критериев, дата клинического диагноза и дата записи в системе хранятся раздельно.",
        "Статус confirmed, probable, ruled_out или unknown и способ подтверждения.",
        "Раннее и позднее начало, тяжесть и органные нарушения кодируются отдельными полями.",
        "Каждый label должен быть воспроизводим из первичных событий либо пройти врачебную adjudication с сохранением версии.",
    ])

    add_section_break(doc)
    add_heading(doc, "7 Преждевременные роды", 1)
    add_paragraph(
        doc,
        "Основные исходы: роды до 37 недель, ранние роды до 32 недель, самопроизвольные роды, роды после преждевременного разрыва оболочек и медицински показанное досрочное родоразрешение. "
        "Общий срок родов без фенотипа недостаточен.",
    )

    add_heading(doc, "Анамнез и фон", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Детализация"],
        [
            ["Предыдущие исходы", "ПР, поздний выкидыш, ПРПО, мертворождение, неонатальная смерть", "Точный срок в днях, механизм начала, многоплодие и показание к родоразрешению"],
            ["Шейка матки", "Конизация, LEEP, ампутация, дилатации, травма, прежний серкляж", "Дата, метод, показание, патология и осложнения"],
            ["Матка", "Аномалии, миомы, операции, рубец", "Тип, локализация и дата"],
            ["Текущая беременность", "Многоплодие, хориальность, ВРТ, интервал между беременностями, кровотечение, предлежание плаценты", "Дата выявления и последующее изменение"],
            ["Образ жизни", "Курение, вещества, низкий или высокий ИМТ, условия труда", "Период и интенсивность"],
            ["Инфекции и воспаление", "ИМП, бактериурия, вагинальные и цервикальные инфекции, системная инфекция", "Материал, тест, результат, лечение и контроль"],
        ],
        [1.25, 3.55, 1.85],
    )

    add_heading(doc, "Симптомы и обследование", 2, page_break_before=True)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Обязательные атрибуты"],
        [
            ["Симптомы", "Схватки, тазовое давление, боль, выделения, кровотечение, подтекание жидкости, снижение шевелений", "Начало, частота, длительность, выраженность и результат осмотра"],
            ["Сократительная активность", "Число схваток, длительность, интервал, токограмма", "Raw сигнал или последовательные записи с временем"],
            ["Осмотр шейки", "Длина влагалищной части, раскрытие, сглаживание, консистенция, положение, предлежащая часть и станция", "Метод осмотра, дата, врачебная роль; каждый повтор отдельно"],
            ["Цервикометрия", "Трансвагинальная длина шейки, минимальное из валидных измерений, раскрытие внутреннего зева, funneling, sludge и оболочки", "Срок, мочевой пузырь, давление датчиком или на дно матки, прибор, оператор, изображения и качество"],
            ["Оболочки", "Целостность, предполагаемое и подтвержденное время ПРПО, цвет и запах вод", "Метод подтверждения, осмотр в зеркалах и тесты"],
            ["Биомаркеры риска ПР", "ИЛ-6 во влагалищном секрете, фосфорилированный IGFBP-1 или ПСИФР-1, PAMG-1; фетальный фибронектин и локальные тесты при выполнении", "Название набора, материал, количественный или качественный результат, порог, партия, загрязнение и время"],
            ["ПРПО", "Скопление или истечение вод при осмотре; симптом папоротника, нитразиновый тест или pH; PAMG-1 и ПСИФР-1 при неясной картине", "Точный метод, материал, результат, порог, время предполагаемого разрыва; кровь, сперма, антисептик и осмотр до теста"],
            ["Воспаление", "ОАК с лейкоцитами и формулой, СРБ; прокальцитонин и другие маркеры при выполнении", "Особенно при ПРПО; температура и КТГ рядом с забором, антибиотики и кортикостероиды до теста"],
            ["СГВ", "ПЦР или культура S. agalactiae из цервикального канала, влагалища и прямой кишки", "Материал и анатомическая точка, метод, дата, чувствительность и антибиотикопрофилактика"],
            ["Мочеполовая инфекция", "Общий анализ и посев мочи; микроскопия и pH влагалищного отделяемого; ПЦР и культура генитальных инфекций при показаниях", "Микроорганизм, количественный результат, чувствительность, дата забора и лечение до или после забора"],
            ["Околоплодные воды, условно", "ИЛ-6, лейкоциты, глюкоза, микроскопия, культура или молекулярный тест; генетические и иные исследования", "Только если выполнен амниоцентез: показание, техника, время, осложнения и полный состав панели"],
            ["Состояние плода", "УЗИ, положение, биометрия, воды, допплер, КТГ", "fetus_id, время и качество"],
        ],
        [1.25, 3.55, 1.85],
        font_size=8.15,
    )

    add_heading(doc, "Вмешательства и исход", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Временная детализация"],
        [
            ["Профилактика", "Вагинальный прогестерон, серкляж, пессарий", "Показание, дата, техника или тип, осложнения, удаление и причина"],
            ["Острый эпизод", "Токолиз, кортикостероиды, магния сульфат для нейропротекции, антибиотики, анальгезия, перевод", "Назначение и фактическое введение, доза, начало, окончание и причина"],
            ["Роды", "Начало регулярной деятельности, разрыв оболочек, полное раскрытие, рождение, плацента", "Точные дата и время, а не только календарный день"],
            ["Фенотип", "Spontaneous PTB, PPROM PTB, indicated PTB, угроза без родов, доношенный исход", "Подтверждение из нескольких источников и статус уверенности"],
            ["Причина indicated PTB", "Преэклампсия, FGR, отслойка, состояние плода, материнская патология и другое", "Основное и дополнительные показания"],
            ["Инфекционные исходы", "Хориоамнионит, эндометрит, неонатальный сепсис, патология плаценты", "Критерии, культура, гистология, дата и лечение"],
        ],
        [1.25, 3.7, 1.7],
    )

    add_section_break(doc)
    add_heading(doc, "8 КТГ и EHG", 1)
    add_paragraph(
        doc,
        "Для сигналов нужны исходные временные ряды. Скан ленты, PDF отчета или только рассчитанные показатели недостаточны для обучения CNN, TCN или SSM. "
        "Если прибор хранит только отчет, он передается отдельно, но отмечается отсутствие raw канала.",
    )

    add_heading(doc, "Raw КТГ", 2)
    add_table(
        doc,
        ["Блок", "Что передать", "Обязательные метаданные"],
        [
            ["Каналы", "ЧСС плода для каждого плода, маточная активность, материнский пульс и дополнительные каналы прибора", "Название канала, единицы, частота дискретизации, шкала и пропуски"],
            ["Файл", "Нативный lossless экспорт и документированный открытый экспорт при возможности", "Формат, версия, контрольная сумма, start_time и end_time"],
            ["Прибор", "Производитель, модель, серийная группа или обезличенный device_id, прошивка и программное обеспечение", "Период эксплуатации, калибровка, фильтры и изменения конфигурации"],
            ["Качество", "Потеря сигнала, артефакты, перестановка датчиков, дублирование материнского пульса, ручные исключения", "Автоматические флаги и экспертная разметка отдельно"],
            ["Разметка", "Акселерации, децелерации, базальный ритм, вариабельность, сокращения, события оператора", "Интервал начала и окончания, автор, версия и уверенность"],
            ["Интерпретация", "Текст и класс по применяемой шкале, автоматический анализ и врачебная оценка", "Название шкалы и версия; время интерпретации и старшинство специалиста"],
        ],
        [1.15, 3.7, 1.8],
        font_size=8.1,
    )

    add_heading(doc, "Контекст КТГ", 2, page_break_before=True)
    add_table(
        doc,
        ["Контекст", "Запрашиваемые данные", "Связь по времени"],
        [
            ["Антенатально", "Срок, показание, движения плода, положение пациентки, прием пищи и лекарства при наличии", "До, во время и после записи"],
            ["В родах", "Период родов, раскрытие, положение плода, воды, температура, АД, материнский пульс", "Ближайшее наблюдение и точное время"],
            ["Воздействия", "Окситоцин, простагландины, эпидуральная анальгезия, вазопрессоры, токолиз, кислород и внутривенная жидкость", "Фактическое введение с началом, изменением дозы и окончанием"],
            ["Действия", "Изменение положения, остановка окситоцина, осмотр, забор крови, решение о родоразрешении", "Не использовать решение врача как единственный label состояния плода"],
            ["Роды", "Время прекращения КТГ, рождения и пережатия пуповины", "Интервал от конца записи до рождения"],
        ],
        [1.2, 3.8, 1.65],
    )

    add_heading(doc, "Исходы для КТГ", 2)
    add_table(
        doc,
        ["Группа", "Показатели", "Обязательная детализация"],
        [
            ["Газы крови", "Артериальная и венозная кровь пуповины, pH, pCO2, pO2, base excess или deficit, бикарбонат, лактат", "Тип сосуда, время забора, время анализа, единицы, анализатор и качество образца"],
            ["Апгар", "Оценка на 1, 5 и 10 минуте и компоненты, если доступны", "Не заменять минимальным или итоговым баллом"],
            ["Реанимация", "Стимуляция, вентиляция маской, интубация, компрессии, лекарства, кислород", "Начало, длительность и ответ"],
            ["Неонатальные исходы", "ОРИТН, гипоксически ишемическая энцефалопатия, судороги, терапевтическая гипотермия, ИВЛ, смерть", "Критерии, дата начала, тяжесть и исход к выписке"],
            ["Родоразрешение", "Способ, срочность и показание", "Отделять клиническое решение от объективного исхода"],
        ],
        [1.25, 3.55, 1.85],
    )

    add_heading(doc, "EHG", 2, page_break_before=True)
    add_table(
        doc,
        ["Блок", "Что передать", "Обязательные метаданные"],
        [
            ["Сигнал", "Все исходные каналы электрогистерографии без предварительного усреднения", "Частота дискретизации, разрядность, диапазон, единицы, start_time и end_time"],
            ["Электроды", "Схема расположения, координаты или фотографический шаблон, reference и ground, тип электродов", "Версия монтажа, подготовка кожи, импеданс по каналам"],
            ["Обработка", "Raw и обработанные версии, фильтры, частоты среза, notch, ресемплинг, удаленные интервалы", "Алгоритм и версия; обработка не должна уничтожать raw файл"],
            ["Синхронизация", "Связь с TOCO, КТГ, клинически отмеченными схватками и осмотром", "Общая временная шкала и известная погрешность синхронизации"],
            ["Контекст", "Срок, симптомы, состояние оболочек, раскрытие, препараты, положение и последующий исход", "События до и после записи"],
            ["Качество", "Движение, отсоединение, электросетевой шум, saturations и исключенные каналы", "Автоматические показатели и экспертная оценка"],
        ],
        [1.2, 3.65, 1.8],
        font_size=8.1,
    )

    add_paragraph(
        doc,
        "Если EHG в медицинской организации не выполняется, это не блокирует первые этапы. В карте источников достаточно отметить not_collected и указать, "
        "есть ли техническая возможность будущего проспективного сбора по отдельному протоколу.",
    )

    add_section_break(doc)
    add_heading(doc, "9 ИЦН", 1)
    add_paragraph(
        doc,
        "На текущем этапе этот блок нужен прежде всего как карточка сбора данных. Один эпизод короткой шейки не должен автоматически считаться подтвержденным диагнозом.",
    )
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Особые требования"],
        [
            ["Анамнез", "Поздние потери, очень ранние ПР, малоболезненное раскрытие, операции и травмы шейки, прежний серкляж", "Точный срок и механизм каждого исхода"],
            ["Серийная цервикометрия", "Длина шейки, раскрытие канала, funneling, пролабирование оболочек, sludge", "Трансвагинальный метод, срок, изображения, прибор, оператор и качество"],
            ["Клинический осмотр", "Симптомы, раскрытие, сглаживание, состояние оболочек, кровотечение, признаки инфекции", "Дата, метод и отрицательные находки"],
            ["До хирургической коррекции", "AB0, RhD и антитела при отсутствии результатов; ОАК, СРБ, температура", "Показание, дата, результат до вмешательства и терапия до забора"],
            ["Влагалищная среда", "Микроскопия мазка и pH влагалищного отделяемого до вмешательства и ежемесячно после серкляжа или установки пессария, если выполнялось", "Материал, метод, флора, лейкоциты, ключевые клетки, дрожжи, pH, лечение и контроль"],
            ["ИППП и дисбиоз", "ПЦР N. gonorrhoeae, T. vaginalis, C. trachomatis, M. genitalium; при нарушении флоры — G. vaginalis, A. vaginae, Lactobacillus spp., U. parvum, U. urealyticum, M. hominis", "Только выполненные панели; состав мишеней, количественный результат, метод и лечение"],
            ["ПРПО", "Осмотр в зеркалах, симптом папоротника, нитразиновый тест или pH, ПСИФР-1 или PAMG-1", "Метод, результат, время предполагаемого разрыва и факторы ложного результата"],
            ["Амниоцентез, условно", "ИЛ-6, лейкоциты, глюкоза, микроскопия, культура и иные исследования околоплодных вод", "Только при выполнении для исключения воспаления или инфекции; показание, срок и осложнения"],
            ["Иные инфекции", "Посев мочи и иные влагалищные, цервикальные или системные исследования", "Забор до антибиотика, отрицательные результаты и время доступности"],
            ["Вмешательства", "Прогестерон, серкляж и техника, пессарий, антибиотики и госпитализация", "Показание, дата, осложнения, удаление и причина"],
            ["Исход", "Срок и механизм родов, ПРПО, хориоамнионит, неонатальный исход", "Подтвержденный фенотип и связь с вмешательствами"],
        ],
        [1.25, 3.65, 1.75],
    )

    add_heading(doc, "10 Задержка роста плода", 1, page_break_before=True)
    add_paragraph(
        doc,
        "Центиль размера и предполагаемой массы рассчитывается детерминированно по выбранному стандарту. ML нужен для анализа траектории роста и риска патологического замедления, а не для замены формулы центилей.",
    )
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Особые требования"],
        [
            ["Датировка", "Опорный срок, ПДР и все версии датировки", "Без надежной датировки траектория роста не интерпретируется"],
            ["Биометрия", "BPD, HC, AC, FL и другие измерения, EFW", "На каждого fetus_id; исходные миллиметры и граммы, формула EFW, прибор, оператор, качество"],
            ["Центили", "Центили каждого показателя и EFW, если рассчитаны", "Название стандарта, версия, популяция и программное обеспечение"],
            ["Траектория", "Все серийные исследования, интервалы и причины внепланового контроля", "Не оставлять только последнее УЗИ"],
            ["Артерия пуповины", "PI, RI, S D, наличие нулевого или реверсного диастолического кровотока", "Сторона или участок измерения, качество спектра и дата"],
            ["Средняя мозговая артерия", "PI, PSV и другие сохраненные параметры", "Сторона, качество, срок"],
            ["CPR", "Готовое значение и компоненты для пересчета", "Формула и версия"],
            ["Маточные артерии", "PI, RI, notch справа и слева", "Сторона, среднее и срок"],
            ["Венозный проток", "PI, a wave и иные параметры", "При наличии и с качеством"],
            ["Воды и плацента", "AFI, максимальный вертикальный карман, локализация, структура, зрелость, инфаркты или отслойка", "Метод, изображения или DICOM и заключение"],
            ["Наблюдение", "КТГ, биофизический профиль, движения плода, госпитализации", "Временная связь с решением о родоразрешении"],
            ["Материнский фон", "Преэклампсия, АГ, курение, заболевания почек, АФС, инфекции и питание при наличии", "Факторы должны предшествовать прогнозу"],
            ["Плацентарные маркеры", "PAPP-A и PlGF из скрининга; PlGF или sFlt-1/PlGF при подозрении и наличии в практике", "Исходное значение, MoM, срок, анализатор и версия диагностического порога"],
            ["Этиология, условно", "TORCH-серология или ПЦР при подозрении на трансплацентарную инфекцию; НИПТ, кариотип, хромосомный микроматричный анализ и иная генетика при выполнении", "Не рутинный скрининг: показание, материал, метод, полный результат и консультация генетика"],
            ["Рождение", "Вес, длина, окружность головы, пол, точный срок, центиль и стандарт", "SGA, low birth weight и FGR остаются разными полями"],
            ["Плацента", "Макроописание, масса, гистология и патологические заключения", "document_id и структурированные признаки при наличии"],
        ],
        [1.15, 3.55, 1.95],
        font_size=8.0,
    )

    add_heading(doc, "11 Гестационный сахарный диабет", 1, page_break_before=True)
    add_paragraph(
        doc,
        "Диагноз по ОГТТ остается пороговым правилом. Исследовательская модель может оценивать риск до теста, поэтому все predictors для ранней модели должны иметь available_at раньше ОГТТ и диагноза.",
    )
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Особые требования"],
        [
            ["Факторы риска", "Возраст, добеременный ИМТ, семейный анамнез диабета, предыдущий ГСД, макросомия, СПКЯ, глюкокортикоиды", "Дата и источник; не включать сведения, появившиеся после диагноза"],
            ["Ранний скрининг", "Глюкоза венозной плазмы натощак; случайная глюкоза при симптомах; HbA1c для подтверждения манифестного диабета, но не для диагностики ГСД", "Лабораторный анализатор; голодание 8-14 часов, материал, фактическое время, единицы и референс"],
            ["ОГТТ", "Идентификатор одной процедуры и глюкоза венозной плазмы 0, 60 и 120 минут после 75 г безводной декстрозы", "Питание до теста, голодание, фактические времена, длительность приема раствора, полнота нагрузки, положение, курение, рвота, острое заболевание и лекарства"],
            ["Преаналитика ОГТТ", "Тип пробирки, ингибитор гликолиза, время до центрифугирования и разделения плазмы, температура хранения", "Причина неполного или недействительного теста; не смешивать с сывороткой, цельной или капиллярной кровью"],
            ["Дифференциальная диагностика", "Повторная глюкоза, HbA1c; GADA, ICA, IAA, IA-2A, ZnT8A, C-пептид и кетоны при выполнении", "Условно при подозрении на манифестный диабет; указать клиническое показание и метод"],
            ["Диагноз", "Дата, критерии и версия рекомендации, тип ГСД, подтверждение", "Не использовать только выписной код без исходных значений"],
            ["Самоконтроль", "Индивидуальные значения глюкозы с привязкой к приему пищи", "До или после еды, время, прибор, единицы и пропуски"],
            ["CGM", "Raw или агрегированный профиль непрерывного мониторинга", "Устройство, сенсор, частота, калибровка, время в диапазоне и правила очистки"],
            ["Лечение", "Диета, обучение, физическая активность, инсулин по типам и дозам, метформин при использовании", "Начало, изменение, фактический прием и приверженность"],
            ["Рост плода", "Серийное УЗИ, AC, EFW, воды и допплер", "fetus_id, формула и стандарт"],
            ["Материнские исходы", "Преэклампсия, индукция, кесарево сечение, травма и послеродовый контроль", "Дата и показание"],
            ["Неонатальные исходы", "Вес и центиль, LGA или макросомия, дистоция плечиков, травма, гипогликемия, дыхательная поддержка, ОРИТН", "Значения глюкозы новорожденного, время и лечение"],
            ["После родов", "ОГТТ 75 г, глюкоза натощак, HbA1c и дальнейший гликемический статус при наличии", "Дата после родов, точки теста, материал, метод, диагноз и последующее наблюдение"],
        ],
        [1.25, 3.6, 1.8],
        font_size=8.05,
    )

    add_heading(doc, "12 Венозные тромбоэмболические осложнения", 1, page_break_before=True)
    add_paragraph(
        doc,
        "Первое назначение блока — воспроизводимая rule based карточка факторов риска и безопасности профилактики. ML рассматривается только после накопления когорты с подтвержденными ТГВ или ТЭЛА, временными факторами, профилактикой и полноценным отрицательным наблюдением.",
    )
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Особые требования"],
        [
            ["Анамнез", "Предыдущие ТГВ или ТЭЛА, обстоятельства события, семейный анамнез, тромбофилия", "Дата, локализация, провоцированный статус и подтверждающее исследование"],
            ["Тромбофилия", "Генетические и лабораторные тесты, АФС и антитела", "Точное название, вариант, генотип или титр, дата, метод; особо чувствительные данные"],
            ["Временные факторы", "Иммобилизация, госпитализация, инфекция, обезвоживание, операция, многоплодие, ожирение, варикоз, гиперстимуляция", "Начало, окончание и тяжесть"],
            ["Роды и послеродовый период", "Кесарево сечение, длительные роды, кровотечение, трансфузия, повторная операция", "Точное время относительно профилактики"],
            ["Лабораторная безопасность", "ОАК с гемоглобином и тромбоцитами, креатинин и расчет клиренса; АЧТВ, ПВ, МНО, фибриноген, печеночные показатели и кровотечение при выполнении", "Время результата относительно антикоагулянта, родов, операции и нейроаксиальной процедуры"],
            ["D-димер", "Все выполненные значения D-димера", "Не считать самостоятельным основным диагностическим признаком при беременности; обязательны срок, метод, референс и DDU или FEU"],
            ["Мониторинг терапии", "Анти-Xa, АЧТВ и тромбоциты при выполнении", "Анти-Xa не является рутинным для профилактических доз: указать показание, препарат, дозу, массу и интервал от введения"],
            ["Профилактика и лечение", "НМГ, НФГ и другие препараты, механическая профилактика", "Показание, препарат, доза, масса тела, назначение, фактическое введение, паузы и причина"],
            ["Подозрение", "Симптомы, клиническая оценка, D димер при выполнении", "Начало симптомов и время каждого теста"],
            ["Верификация", "Компрессионное УЗИ вен, КТ ангиография легочных артерий, вентиляционно перфузионное исследование, эхокардиография", "Сторона, сегмент, заключение, дата, изображения или идентификатор исследования"],
            ["Исход", "Подтвержденный ТГВ, ТЭЛА, поверхностный тромбофлебит, кровотечение, реанимация, смерть", "Статус подтверждения и срок отрицательного наблюдения"],
        ],
        [1.25, 3.55, 1.85],
        font_size=8.05,
    )

    add_section_break(doc)
    add_heading(doc, "13 Материнские и неонатальные исходы", 1)
    add_paragraph(
        doc,
        "Исходы передаются независимо от того, какая модель строится первой. Это защищает от повторных выгрузок и позволяет в будущем пересматривать фенотипы без потери первичных фактов.",
    )
    add_heading(doc, "Роды и материнский исход", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Детализация"],
        [
            ["Роды", "Дата и время, точный GA, начало спонтанное или индуцированное, способ родоразрешения", "Срочность, основное и дополнительные показания"],
            ["Ход родов", "Разрыв оболочек, воды, длительность периодов, окситоцин, обезболивание, температура и инфекции", "Точные события на общей временной шкале"],
            ["Операция", "Тип, разрез, анестезия, осложнения, решение и начало операции", "Интервалы decision to incision и decision to birth при наличии"],
            ["Кровопотеря", "Оценка или измерение, послеродовое кровотечение, препараты, операции и трансфузия", "Метод оценки, компоненты крови, объем и время"],
            ["Тяжелые исходы", "ОРИТ, ИВЛ, ОПП, инсульт, сепсис, повторная операция, реадмиссия и материнская смерть", "Дата начала, критерии, длительность и итог"],
            ["Послеродовый период", "АД, лаборатории, осложнения, лекарства, повторные обращения", "До 42 дней после родов при доступности"],
        ],
        [1.25, 3.65, 1.75],
    )

    add_heading(doc, "Исход плода и новорожденного", 2)
    add_table(
        doc,
        ["Блок", "Запрашиваемые данные", "Детализация"],
        [
            ["Статус", "Живорождение, антенатальная или интранатальная гибель, время рождения", "На каждого fetus_id отдельно"],
            ["Антропометрия", "Пол, масса, длина, окружность головы, центиль, SGA и LGA", "Стандарт и версия расчета"],
            ["Адаптация", "Апгар на 1, 5 и 10 минуте, температура, дыхание и сатурация", "Компоненты шкалы при наличии"],
            ["Газы пуповины", "Артериальный и венозный pH, pCO2, pO2, base excess или deficit, лактат", "Сосуд, время, единицы, анализатор и качество"],
            ["Лаборатория новорожденного", "Глюкоза, билирубин и фракции, ОАК, гематокрит, тромбоциты, СРБ, прокальцитонин, посевы, электролиты и иные выполненные анализы", "Материал, время от рождения, показание, лечение до забора и все отрицательные результаты"],
            ["Иммуногематология новорожденного", "AB0, RhD, прямая проба Кумбса и иные тесты при выполнении", "Связь с материнскими антителами, фототерапией, трансфузией и гемолизом"],
            ["Реанимация", "Вентиляция, интубация, компрессии, лекарства, кислород", "Время начала, длительность и ответ"],
            ["ОРИТН", "Поступление, длительность, респираторная поддержка, питание и основные процедуры", "Дата и время каждого изменения"],
            ["Заболеваемость", "РДС, БЛД, ВЖК, НЭК, сепсис, ГИЭ, судороги, гипогликемия, ретинопатия", "Критерии, степень, дата и подтверждение"],
            ["Смерть", "Неонатальная смерть, дата и причина", "До 7 и 28 дней либо до выписки при наличии"],
            ["Плацента", "Масса, макроскопия, гистология, инфаркты, воспаление и сосудистая патология", "document_id и структурированные признаки"],
        ],
        [1.25, 3.6, 1.8],
        font_size=8.1,
    )

    add_heading(doc, "14 Дополнительный слой для SSM", 1)
    add_paragraph(
        doc,
        "SSM модель требует не другого списка диагнозов, а более точной последовательности событий и большего числа сырых модальностей. "
        "Этот слой запрашивается после проверки простых baseline моделей и качества трех приоритетных задач.",
    )
    add_table(
        doc,
        ["Требование", "Дополнительные данные", "Критическая причина"],
        [
            ["Полная последовательность", "Все визиты, результаты, назначения, введения, документы и сигналы, включая нормальные и отрицательные", "SSM изучает переходы состояния, а не только итоговую сводку"],
            ["Двойное время", "observed_at и available_at, а также order, specimen, result и verify при наличии", "Исключение утечки будущей информации"],
            ["Линия лабораторной пробы", "order_id, panel_id, accession_id, specimen_id, отдельные аналиты, add-on, разведение, повтор, предварительный, финальный и исправленный статус", "Сохраняет причинный порядок и не превращает исправленный результат в знание из прошлого"],
            ["Несколько беременностей", "Связанные эпизоды одной женщины и довременной анамнез", "Оценка устойчивого фона без смешивания исходов"],
            ["Тексты", "Обезличенные записи врача, жалобы, дневники, выписки, протоколы УЗИ и операции", "Сохранять document_id, время, роль автора и исходный фрагмент"],
            ["Изображения", "DICOM УЗИ и допплера, ключевые кадры и таблицы измерений", "UID после безопасного переназначения, прибор и связь с fetus_id"],
            ["Raw сигналы", "КТГ, TOCO, EHG и материнский пульс", "Общая временная шкала, частота и маска качества"],
            ["Лечение как экспозиция", "Заказ, выдача, введение, изменение дозы и отмена", "Назначение не равно фактическому получению"],
            ["План и пропуски", "Ожидаемое окно обследования, выполнено или нет и причина", "Пропуск из за срока, решения врача и отсутствия данных имеет разный смысл"],
            ["Версии", "Исправленные результаты, отмененные документы, смена датировки и локальных правил", "История не должна переписываться задним числом"],
            ["Системные изменения", "Смена МИС, лаборатории, прибора, реагента, протокола и подразделения", "Контроль domain shift"],
            ["Маски модальностей", "Доступность каждой группы данных на каждом моменте", "Модель должна отличать не выполнено от не выгружено"],
            ["Проспективный поток", "Время появления события в интеграции и задержка доставки", "Подготовка будущего silent режима"],
        ],
        [1.4, 3.4, 1.85],
        font_size=8.05,
    )

    add_heading(doc, "Показатели готовности к SSM", 2)
    add_bullets(doc, [
        "Распределение длины последовательности на беременность по каждой модальности.",
        "Доля событий с известным available_at и точным временем, а не только датой.",
        "Доля беременностей с исходом, серийными наблюдениями и документированной причиной пропусков.",
        "Число положительных исходов по каждому целевому горизонту и календарному периоду.",
        "Доля сигналов, связанных с fetus_id и исходом, и распределение signal loss.",
        "Даты смены источников, приборов и протоколов для временной и внешней проверки.",
    ])

    add_section_break(doc)
    add_heading(doc, "15 Форматы и структура поставки", 1)
    add_table(
        doc,
        ["Тип данных", "Предпочтительный формат", "Минимальные требования"],
        [
            ["Структурированные таблицы", "Parquet; допустим CSV UTF 8", "Одна строка на событие, схема, типы, словарь и контрольная сумма"],
            ["Справочники", "CSV или XLSX", "Локальный код, название, период действия и нормализованное соответствие при наличии"],
            ["КТГ и EHG", "Нативный формат плюс WFDB, EDF, HDF5 или документированный бинарный экспорт", "Каналы, частота, единицы, время, device_id и файл метаданных"],
            ["Изображения", "DICOM", "Безопасное переназначение UID, сохранение технических тегов, удаление прямых идентификаторов"],
            ["Документы", "Обезличенный текст UTF 8 или JSON; PDF при необходимости", "document_id, тип, дата, версия, качество OCR и связь с событиями"],
            ["Манифест", "JSON и CSV", "snapshot_id, extract_version, фильтры, файлы, строки, байты, SHA 256 и дата формирования"],
        ],
        [1.4, 2.0, 3.25],
    )

    add_heading(doc, "Правила представления", 2)
    add_bullets(doc, [
        "Дата и время в ISO 8601 с часовым поясом или с явно указанной локальной зоной медицинской организации.",
        "Дробная часть числа передается числом, а не строкой. Исходное строковое значение сохраняется отдельно, если возможно.",
        "Единицы не угадываются. Передается исходная единица и, при нормализации, алгоритм и нормализованная единица.",
        "Пустое значение, ноль, отрицательный результат, ниже предела определения и не выполнено кодируются по разному.",
        "Локальные коды анализов и услуг сохраняются. Маппинг на LOINC, ATC, МНН или другую терминологию передается при наличии, но не заменяет локальный код.",
        "МКБ 10 передается вместе с текстом диагноза, статусом и датой. Выписной код не заменяет исходные критерии phenotype.",
        "Файлы и строки связываются только через устойчивые псевдонимные идентификаторы.",
    ])

    add_heading(doc, "16 Обезличивание и правовой контур", 1)
    add_paragraph(
        doc,
        "Конкретное правовое основание, необходимость согласия, решение локального этического комитета, состав договора и режим доступа определяет медицинская организация "
        "совместно с юридической службой, ответственным за защиту персональных данных и информационной безопасностью. Этот документ описывает состав данных и не заменяет такое решение.",
    )
    add_table(
        doc,
        ["Требование", "Практическое правило"],
        [
            ["Псевдонимизация", "Ключ соответствия остается у медицинской организации. Исследователь получает стабильные person_id, pregnancy_id и fetus_id без исходных номеров."],
            ["Прямые идентификаторы", "Не передавать ФИО, адрес, телефон, электронную почту, СНИЛС, полис, паспорт, номера карт и иные прямые идентификаторы."],
            ["Косвенные идентификаторы", "Огрублять редкие географические, профессиональные и демографические комбинации по результатам оценки риска повторной идентификации."],
            ["Даты", "В защищенном контуре сохранять точные времена. При внешней передаче применять согласованный постоянный сдвиг, который сохраняет интервалы и срок беременности."],
            ["Свободный текст", "Удалять ФИО, контакты, адреса, номера документов, имена родственников и иные идентификаторы; сохранять качество и журнал правил деидентификации."],
            ["DICOM и сигналы", "Удалять идентификаторы из заголовков, burned in текста и имен файлов; проверять приватные теги и UID."],
            ["Доступ", "Минимально необходимые роли, журналирование, шифрование при хранении и передаче, ограничение копирования и согласованный срок хранения."],
            ["Публикация", "Не публиковать строки, редкие сочетания или изображения, которые позволяют повторную идентификацию. Публичные результаты только агрегированные."],
            ["Удаление", "Определить окончание проекта, возврат или уничтожение копий, подтверждение удаления и порядок работы с резервными копиями."],
        ],
        [1.55, 5.1],
        font_size=8.3,
    )

    add_heading(doc, "17 Контроль качества и приемка", 1)
    add_paragraph(
        doc,
        "Поставка принимается после автоматических проверок и совместной клинической сверки. Ошибки не исправляются молча: источник предоставляет новую версию и журнал изменений.",
    )
    add_table(
        doc,
        ["Проверка", "Что контролируется", "Выход"],
        [
            ["Объем", "Строки, уникальные женщины, беременности, плоды, исходы и файлы", "Сверка с манифестом и внутренней статистикой организации"],
            ["Связность", "Все внешние ключи, одна беременность на событие, fetus_id для плодовых данных", "Список потерянных и конфликтных ссылок"],
            ["Дубли", "Полные и смысловые дубли событий, повторные импорты документов и сигналов", "Правило выбора или сохранения версий"],
            ["Время", "Порядок событий, observed_at и available_at, срок беременности, дата родов", "Невозможные последовательности и неопределенность"],
            ["Единицы", "Смешение единиц, смена анализаторов и референсов, невозможные диапазоны", "Таблица преобразований и флаги качества"],
            ["Пропуски", "Доля по полю, году, подразделению, исходу и источнику", "Матрица missingness и причины отсутствия"],
            ["Labels", "Воспроизводимость исхода из первичных событий и отсутствие утечки", "Версия label contract и список adjudication cases"],
            ["Сигналы", "Декодирование, длительность, частота, gaps, signal loss, связь с исходом", "Signal quality report по прибору и году"],
            ["Обезличивание", "Прямые и косвенные идентификаторы в таблицах, текстах, DICOM, файлах и путях", "Акт проверки и перечень остаточных ограничений"],
            ["Клиническая сверка", "100–200 историй из разных лет, исходов и источников", "Протокол расхождений и согласованная версия исправлений"],
        ],
        [1.25, 3.55, 1.85],
        font_size=8.15,
    )

    add_heading(doc, "Запреты для модельного набора", 2)
    add_bullets(doc, [
        "Не включать в predictors сведения, ставшие доступными после index time.",
        "Не использовать срок и массу при рождении как вход для дородового прогноза.",
        "Не разделять разные окна одной КТГ или EHG записи между train и test.",
        "Не разделять беременности одной женщины между train и test.",
        "Не считать отсутствующий анализ нормальным или отрицательным.",
        "Не смешивать spontaneous и indicated PTB, SGA и FGR, экспертный класс КТГ и объективный неонатальный исход.",
        "Не записывать восстановленное моделью значение как фактически измеренное.",
    ])

    add_heading(doc, "18 Пакет первого согласования", 1)
    add_paragraph(
        doc,
        "Для начала работ достаточно согласовать и передать следующий минимальный пакет. Он не ограничивает целевой набор и используется для проверки реализуемости.",
    )
    add_numbered(doc, [
        "Карту всех систем источников и ответственных подразделений.",
        "Словарь полей и локальных кодов с единицами, референсами и периодом действия.",
        "Агрегированные объемы и числа исходов по годам без передачи персональных данных.",
        "Пилот 200–500 последовательных беременностей со стабильными person_id, pregnancy_id и fetus_id.",
        "Для пилота — визиты, давление, основные лаборатории, лекарства, УЗИ, госпитализации, роды и исходы новорожденных.",
        "Не менее нескольких примеров raw КТГ каждого типа прибора с документацией формата; EHG — только если реально собирается.",
        "Манифест выгрузки, описание обезличивания, контрольные суммы и список известных ограничений.",
        "Совместную клиническую проверку тестовых историй до формирования полной когорты.",
    ])
    add_paragraph(
        doc,
        "После приемки пилота первым полноценным релизом должна стать последовательная структурированная когорта для преэклампсии и преждевременных родов. "
        "Raw КТГ может идти отдельным параллельным пакетом, если архив приборов технически готов раньше или позже МИС и ЛИС.",
    )

    add_section_break(doc)
    add_heading(doc, "Приложение А Логические таблицы", 1)
    add_table(
        doc,
        ["Таблица", "Зерно", "Ключевые поля"],
        [
            ["person", "Одна женщина", "person_id, демография, разрешенные устойчивые признаки"],
            ["pregnancy", "Одна беременность", "pregnancy_id, person_id, dating, EDD, plurality, outcome_status"],
            ["fetus", "Один плод", "fetus_id, pregnancy_id, label, chorionicity, amnionicity"],
            ["encounter", "Один контакт", "encounter_id, pregnancy_id, start, end, setting, department, reason"],
            ["clinical_event", "Один факт", "event_id, concept, value, unit, observed_at, available_at, source, version"],
            ["vital_sign", "Одно измерение", "BP, pulse, temperature, SpO2, posture, arm, device, repeat_no"],
            ["laboratory", "Один аналит", "accession, specimen, analyte, result, unit, reference, method, analyzer"],
            ["diagnosis", "Одна версия диагноза", "code, text, status, onset, recorded_at, confirmed_at, source"],
            ["medication_order", "Одно назначение", "INN, dose, route, frequency, start, stop, indication"],
            ["medication_administration", "Одно введение", "drug, dose, route, administered_at, status"],
            ["procedure", "Одна процедура", "type, indication, start, end, result, complication"],
            ["ultrasound", "Одно исследование одного плода", "fetus_id, GA, measurements, EFW formula, device, report_id"],
            ["doppler", "Один сосуд или сторона", "fetus_id, vessel, side, PI, RI, S D, waveform quality"],
            ["ctg_recording", "Одна запись", "recording_id, fetus_id, start, end, device, raw_file, quality"],
            ["ehg_recording", "Одна запись", "recording_id, pregnancy_id, montage, channels, raw_file, quality"],
            ["document", "Одна версия документа", "document_id, type, created_at, signed_at, author_role, text, version"],
            ["care_plan_task", "Одна задача наблюдения", "window, condition, status, completed_event_id, missing_reason"],
            ["maternal_outcome", "Один исход матери", "definition_id, onset, confirmed_at, severity, source, adjudication"],
            ["neonatal_outcome", "Один исход ребенка", "fetus_id, definition_id, onset, severity, source, adjudication"],
            ["extract_manifest", "Один файл или таблица", "snapshot_id, extract_version, rows, bytes, SHA 256, filters"],
        ],
        [1.55, 1.75, 3.35],
        font_size=8.0,
    )

    add_heading(doc, "Приложение Б Справочники и единицы", 1)
    add_table(
        doc,
        ["Объект", "Что сохранить", "Примечание"],
        [
            ["Лаборатория", "Локальный код, название, материал, метод, единица, референс, analyzer и период действия", "Маппинг на LOINC при наличии"],
            ["Лекарства", "Локальный код, МНН, торговое название, форма, концентрация и ATC при наличии", "Сохранять фактическую дозу и путь"],
            ["Диагнозы", "МКБ 10 и локальные расширения, текст, статус", "Версия классификатора и период действия"],
            ["Услуги", "Локальный код и код номенклатуры медицинских услуг при наличии", "Название и подразделение"],
            ["УЗИ", "Параметры, формулы EFW, стандарты центилей и версии ПО", "Не пересчитывать молча"],
            ["КТГ", "Шкалы интерпретации, версии автоматического анализа и device mappings", "Raw сигнал остается первичным"],
            ["Подразделения", "Укрупненный обезличенный department_id, профиль и уровень помощи", "Без ФИО персонала"],
            ["Причины пропуска", "not_due, clinician_decision, patient_declined, unavailable, lost_to_followup, data_error", "Локальные значения маппируются без потери исходника"],
        ],
        [1.25, 3.5, 1.9],
        font_size=8.15,
    )

    add_heading(doc, "Приложение В Источники", 1)
    add_paragraph(
        doc,
        "Состав полей сверялся с внутренней спецификацией Pregnancy State Platform версии 3.0 и с действующими на дату документа российскими клиническими и правовыми источниками. "
        "При подписании соглашения медицинская организация фиксирует актуальные версии клинических рекомендаций и локальных регламентов.",
    )

    add_source(doc, "Pregnancy State Platform", "Финальная клиническо техническая спецификация, версия 3.0 от 21 сентября 2026 года")
    add_source(doc, "Рубрикатор клинических рекомендаций Минздрава России", "Официальный каталог и актуальные версии документов", "https://cr.minzdrav.gov.ru/")
    add_source(doc, "Преэклампсия и гипертензивные расстройства", "Клинические рекомендации Минздрава России, 2024, ID 637, версия 2", "https://cr.minzdrav.gov.ru/view-cr/637_2")
    add_source(doc, "Преждевременные роды", "Клинические рекомендации Минздрава России, 2024, ID 331, версия 2", "https://cr.minzdrav.gov.ru/view-cr/331_2")
    add_source(doc, "Истмико-цервикальная недостаточность", "Клинические рекомендации Минздрава России, 2024, ID 671", "https://cr.minzdrav.gov.ru/view-cr/671_2")
    add_source(doc, "Гестационный сахарный диабет", "Клинические рекомендации Минздрава России, 2024, ID 841", "https://cr.minzdrav.gov.ru/view-cr/841_1")
    add_source(doc, "Недостаточный рост плода", "Клинические рекомендации Минздрава России, 2025, ID 722", "https://cr.minzdrav.gov.ru/view-cr/722_2")
    add_source(doc, "Венозные осложнения во время беременности и послеродовом периоде", "Клинические рекомендации Минздрава России, актуальная версия ID 723", "https://cr.minzdrav.gov.ru/view-cr/723_2")
    add_source(doc, "Нормальная беременность", "Клинические рекомендации Минздрава России, ID 288", "https://cr.minzdrav.gov.ru/view-cr/288_2")
    add_source(doc, "Нормальные роды", "Клинические рекомендации Минздрава России, 2024, ID 636", "https://cr.minzdrav.gov.ru/view-cr/636_2")
    add_source(doc, "Порядок оказания медицинской помощи по профилю акушерство и гинекология", "Приказ Минздрава России от 19 декабря 2025 года № 747н", "https://publication.pravo.gov.ru/document/0001202512300089")
    add_source(doc, "Об основах охраны здоровья граждан в Российской Федерации", "Федеральный закон № 323-ФЗ, включая требования о врачебной тайне", "https://minzdrav.gov.ru/documents/7025-")
    add_source(doc, "О персональных данных", "Федеральный закон № 152-ФЗ", "https://mintrud.gov.ru/docs/laws/130")
    add_source(doc, "Подсистема обезличивания персональных данных", "Описание Минздрава России", "https://minzdrav.gov.ru/special/ministry/web-site/informatsionnye-sistemy-minzdrava-rossii/katalog-podsistem-egisz/podsistema-obezlichivaniya-personalnyh-dannyh")

    add_paragraph(
        doc,
        "Дата актуализации источников 23 сентября 2026 года. Перед началом клинической валидации версии проверяются повторно.",
        after=0,
    )

    return doc


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
